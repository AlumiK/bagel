import torch
import bagel
import numpy as np

from typing import Sequence, Dict, Optional
from torch.utils.data import DataLoader
from torch.backends import cudnn


class AutoencoderLayer(torch.nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self._hidden = torch.nn.Sequential()
        last_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self._hidden.add_module(f'hidden_{i}', torch.nn.Linear(last_dim, hidden_dim))
            self._hidden.add_module(f'relu_{i}', torch.nn.ReLU())
            last_dim = hidden_dim
        self._mean = torch.nn.Linear(last_dim, output_dim)
        self._std = torch.nn.Sequential(
            torch.nn.Linear(last_dim, output_dim),
            torch.nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        x = self._hidden(x)
        mean = self._mean(x)
        std = self._std(x) + 1e-6
        return mean, std


class ConditionalVariationalAutoencoder(torch.nn.Module):

    def __init__(self, encoder: AutoencoderLayer, decoder: AutoencoderLayer, device: str):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._device = device

    def forward(self, inputs, **kwargs):
        x, y = tuple(inputs)
        n_samples = kwargs.get('n_samples', 1)
        concatted = torch.cat([x, y], dim=-1)
        z_mean, z_std = self._encoder(concatted)
        q_zx = torch.distributions.Normal(z_mean, z_std)
        p_z = torch.distributions.Normal(
            torch.zeros(z_mean.size()).to(self._device),
            torch.ones(z_std.size()).to(self._device)
        )
        z = p_z.sample((n_samples,)) * torch.unsqueeze(z_std, 0) + torch.unsqueeze(z_mean, 0)
        y = y.expand(n_samples, -1, -1)
        concatted = torch.cat([z, y], dim=-1)
        x_mean, x_std = self._decoder(concatted)
        p_xz = torch.distributions.Normal(x_mean, x_std)
        return q_zx, p_xz, z


class Bagel:

    def __init__(self,
                 window_size: int = 120,
                 hidden_dims: Sequence = (100, 100),
                 latent_dim: int = 8,
                 learning_rate: float = 1e-3,
                 dropout_rate: float = 0.1,
                 device: Optional[str] = None):
        cudnn.benchmark = True
        if device is None:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device

        self._window_size = window_size
        self._cond_size = 60 + 24 + 7
        self._dropout_rate = dropout_rate
        self._model = ConditionalVariationalAutoencoder(
            encoder=AutoencoderLayer(
                input_dim=self._window_size + self._cond_size,
                output_dim=latent_dim,
                hidden_dims=hidden_dims
            ),
            decoder=AutoencoderLayer(
                input_dim=latent_dim + self._cond_size,
                output_dim=self._window_size,
                hidden_dims=list(reversed(hidden_dims))
            ),
            device=self._device
        ).to(self._device)
        self._p_z = torch.distributions.Normal(
            torch.zeros(latent_dim).to(self._device),
            torch.ones(latent_dim).to(self._device)
        )
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate, weight_decay=1e-3)
        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=10, gamma=0.75)

    @staticmethod
    def _m_elbo(x: torch.Tensor,
                z: torch.Tensor,
                normal: torch.Tensor,
                q_zx: torch.distributions.Normal,
                p_z: torch.distributions.Normal,
                p_xz: torch.distributions.Normal) -> torch.Tensor:
        x = torch.unsqueeze(x, 0)
        normal = torch.unsqueeze(normal, 0)
        log_p_xz = p_xz.log_prob(x)
        log_q_zx = torch.sum(q_zx.log_prob(z), -1)
        log_p_z = torch.sum(p_z.log_prob(z), -1)
        ratio = (torch.sum(normal, -1) / float(normal.size()[-1]))
        return torch.mean(torch.sum(log_p_xz * normal, -1) + log_p_z * ratio - log_q_zx)

    def _missing_imputation(self,
                            x: torch.Tensor,
                            y: torch.Tensor,
                            normal: torch.Tensor,
                            max_iter: int = 10) -> torch.Tensor:
        with torch.no_grad():
            for _ in range(max_iter):
                _, p_xz, _ = self._model([x, y])
                x[normal == 0.] = p_xz.sample()[0][normal == 0.]
        return x

    def fit(self,
            kpi: bagel.data.KPI,
            epochs: int,
            validation_kpi: Optional[bagel.data.KPI] = None,
            batch_size: int = 256,
            verbose: int = 1) -> Dict:
        dataset = bagel.data.KPIDataset(kpi, window_size=self._window_size, missing_injection_rate=0.01)
        dataset = DataLoader(dataset.to_torch(self._device), batch_size=batch_size, shuffle=True, drop_last=True)
        validation_dataset = None
        if validation_kpi is not None:
            validation_dataset = bagel.data.KPIDataset(validation_kpi, window_size=self._window_size)
            validation_dataset = DataLoader(validation_dataset.to_torch(self._device),
                                            batch_size=batch_size,
                                            shuffle=True)

        losses = []
        val_losses = []
        history = {}
        progbar = None
        if verbose == 1:
            print('Training Epoch')
            progbar = bagel.utils.Progbar(epochs,
                                          interval=0.5,
                                          stateful_metrics=['loss', 'val_loss'],
                                          unit_name='epoch')

        for epoch in range(epochs):
            epoch_losses = []
            epoch_val_losses = []
            epoch_val_loss = np.nan

            if verbose == 2:
                print(f'Training Epoch {epoch + 1}/{epochs}')
                progbar = bagel.utils.Progbar(
                    target=len(dataset) + (0 if validation_kpi is None else len(validation_dataset)),
                    interval=0.5
                )
            self._model.train()
            for batch in dataset:
                self._optimizer.zero_grad()
                x, y, normal = batch
                y = torch.nn.Dropout(self._dropout_rate)(y)
                q_zx, p_xz, z = self._model([x, y])
                loss = -self._m_elbo(x, z, normal, q_zx, self._p_z, p_xz)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=10.)
                self._optimizer.step()
                epoch_losses.append(loss)
                if verbose == 2:
                    progbar.add(1, values=[('loss', loss.detach().cpu().numpy())])
            epoch_loss = torch.mean(torch.as_tensor(epoch_losses)).numpy()
            losses.append(epoch_loss)

            if validation_kpi is not None:
                with torch.no_grad():
                    self._model.eval()
                    for batch in validation_dataset:
                        x, y, normal = batch
                        q_zx, p_xz, z = self._model([x, y])
                        val_loss = -self._m_elbo(x, z, normal, q_zx, self._p_z, p_xz)
                        epoch_val_losses.append(val_loss)
                        if verbose == 2:
                            progbar.add(1, values=[('val_loss', val_loss.cpu().numpy())])
                epoch_val_loss = torch.mean(torch.as_tensor(epoch_val_losses)).numpy()
                val_losses.append(epoch_val_loss)

            if verbose == 1:
                values = []
                if not np.isnan(epoch_loss):
                    values.append(('loss', epoch_loss))
                if not np.isnan(epoch_val_loss):
                    values.append(('val_loss', epoch_val_loss))
                progbar.add(1, values=values)

            self._lr_scheduler.step()

        history['loss'] = losses
        if len(val_losses) > 0:
            history['val_loss'] = val_losses
        return history

    def predict(self, kpi: bagel.data.KPI, batch_size: int = 256, verbose: int = 1) -> np.ndarray:
        print('Testing Epoch')
        kpi = kpi.no_labels()
        dataset = bagel.data.KPIDataset(kpi, window_size=self._window_size)
        dataset = DataLoader(dataset.to_torch(self._device), batch_size=batch_size)
        progbar = None
        if verbose == 1:
            progbar = bagel.utils.Progbar(len(dataset), interval=0.5)
        anomaly_scores = []
        with torch.no_grad():
            self._model.eval()
            for batch in dataset:
                x, y, normal = batch
                x = self._missing_imputation(x, y, normal)
                q_zx, p_xz, z = self._model([x, y], n_samples=128)
                test_loss = -self._m_elbo(x, z, normal, q_zx, self._p_z, p_xz)
                log_p_xz = p_xz.log_prob(x)
                anomaly_scores.extend(-torch.mean(log_p_xz[:, :, -1], dim=0))
                if verbose == 1:
                    progbar.add(1, values=[('test_loss', test_loss.cpu().numpy())])
        anomaly_scores = np.asarray(anomaly_scores, dtype=np.float32)
        return np.concatenate([np.ones(self._window_size - 1) * np.min(anomaly_scores), anomaly_scores])

    def save(self, path: str):
        torch.save(self._model.state_dict(), path)

    def load(self, path: str):
        self._model.load_state_dict(torch.load(path))

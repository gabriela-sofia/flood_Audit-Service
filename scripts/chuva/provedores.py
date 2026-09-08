"""Provider boundary and replayable cache for rain acquisition."""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contrato_chuva import (
    METHOD_VERSION, PRODUCT_ID, PROVIDER_ID, PROVIDER_URL, TIMEZONE,
    derive_rain_features, quality_control,
)


@dataclass(frozen=True)
class RainRequest:
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    timezone: str = TIMEZONE

    def validated(self) -> "RainRequest":
        if not -90 <= float(self.latitude) <= 90:
            raise ValueError("latitude outside [-90, 90]")
        if not -180 <= float(self.longitude) <= 180:
            raise ValueError("longitude outside [-180, 180]")
        if self.start_date > self.end_date:
            raise ValueError("inverted rain period")
        return self


@dataclass(frozen=True)
class RainAcquisition:
    request: RainRequest
    raw_payload: dict[str, Any]
    raw_sha256: str
    acquired_at: str
    cache_key: str
    cache_mode: str


class RainProvider(ABC):
    provider_id: str
    product_id: str

    @abstractmethod
    def acquire(self, request: RainRequest, *, online: bool,
                horario: bool = False) -> RainAcquisition:
        raise NotImplementedError

    @abstractmethod
    def derive(self, acquisition: RainAcquisition) -> dict:
        raise NotImplementedError


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


class OpenMeteoEra5LandProvider(RainProvider):
    provider_id = PROVIDER_ID
    product_id = PRODUCT_ID

    def __init__(self, cache_dir: Path, *, session=None, timeout_seconds: float = 30,
                 retries: int = 3, backoff_seconds: float = 0.25,
                 min_interval_seconds: float = 0.0,
                 espera_no_limite: float = 20.0):
        self.cache_dir = Path(cache_dir)
        self.session = session
        self.timeout_seconds = float(timeout_seconds)
        self.retries = int(retries)
        self.backoff_seconds = float(backoff_seconds)
        self.min_interval_seconds = float(min_interval_seconds)
        self.espera_no_limite = float(espera_no_limite)
        self._last_request_at = 0.0

    def _key(self, request: RainRequest, *, horario: bool = False) -> str:
        """Chave do cache. O namespace horario so entra QUANDO pedido.

        Se `horario` entrasse sempre no corpo, a chave de toda requisicao
        diaria mudaria e os 92 mil payloads ja em cache virariam orfaos -- o
        replay cientifico do projeto pararia de reproduzir. Por isso a chave
        diaria fica byte a byte identica a que sempre foi.
        """
        body = {
            "provider": self.provider_id, "product": self.product_id,
            "method_version": METHOD_VERSION, **asdict(request),
        }
        if horario:
            body["variables"] = "hourly:precipitation"
        return hashlib.sha256(_canonical_json(body)).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def acquire(self, request: RainRequest, *, online: bool,
                horario: bool = False) -> RainAcquisition:
        """`horario=True` pede a serie de hora em hora em vez do acumulado do dia.

        O acumulado nao distingue 80 mm em duas horas de 80 mm no dia inteiro, e
        para alagamento urbano a intensidade e o mecanismo: a agua aparece quando
        a taxa de chegada excede a de escoamento da galeria.
        """
        request = request.validated()
        key = self._key(request, horario=horario)
        path = self._path(key)
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            payload = cached["raw_payload"]
            raw_hash = hashlib.sha256(_canonical_json(payload)).hexdigest()
            if raw_hash != cached["raw_sha256"]:
                raise RuntimeError(f"rain cache hash mismatch: {path}")
            return RainAcquisition(request, payload, raw_hash,
                                   cached["acquired_at"], key, "replay")
        if not online:
            raise FileNotFoundError(
                f"rain cache miss for {key}; scientific replay does not access network")

        if self.session is None:
            import requests
            self.session = requests.Session()
        params = {
            "latitude": request.latitude, "longitude": request.longitude,
            "start_date": request.start_date, "end_date": request.end_date,
            "timezone": request.timezone,
        }
        params["hourly" if horario else "daily"] = (
            "precipitation" if horario else "precipitation_sum")
        last_error = None
        for attempt in range(self.retries + 1):
            wait = self.min_interval_seconds - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.session.get(PROVIDER_URL, params=params,
                                            timeout=self.timeout_seconds)
                self._last_request_at = time.monotonic()
                if response.status_code == 429:
                    # Limite de uso do provedor. O backoff exponencial curto
                    # (0,25 s) e para falha de rede, e aqui ele so devolve o
                    # pedido a fila mais rapido do que a janela do limite: a
                    # corrida do pico horario chegou a 29% de falha assim.
                    # Esperar a janela e a unica resposta correta -- e educada
                    # com uma API publica e gratuita.
                    time.sleep(self.espera_no_limite * (attempt + 1))
                    raise RuntimeError("HTTP 429 (limite do provedor)")
                if response.status_code >= 500:
                    raise RuntimeError(f"HTTP {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("a resposta do provedor nao e um objeto")
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.retries:
                    raise RuntimeError(
                        f"provedor de chuva falhou apos {attempt + 1} tentativas: {exc}") from exc
                time.sleep(self.backoff_seconds * 2 ** attempt)
        else:  # pragma: no cover
            raise RuntimeError(str(last_error))

        raw_hash = hashlib.sha256(_canonical_json(payload)).hexdigest()
        acquired_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record = {
            "provider": self.provider_id, "product": self.product_id,
            "method_version": METHOD_VERSION, "request": asdict(request),
            "acquired_at": acquired_at, "raw_sha256": raw_hash,
            "raw_payload": payload,
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return RainAcquisition(request, payload, raw_hash, acquired_at, key, "online")

    def derive(self, acquisition: RainAcquisition) -> dict:
        daily = acquisition.raw_payload.get("daily") or {}
        dates = daily.get("time") or []
        values = daily.get("precipitation_sum") or []
        qc = quality_control(dates, values,
                             expected_start=acquisition.request.start_date,
                             expected_end=acquisition.request.end_date)
        if not qc["ok"]:
            raise ValueError(f"rain quality control failed: {qc['issues']}")
        derived = derive_rain_features(values, expected_days=len(dates))
        return {
            **derived,
            "provider": self.provider_id,
            "product": self.product_id,
            "timezone": acquisition.request.timezone,
            "reference_period": {
                "start": acquisition.request.start_date,
                "end": acquisition.request.end_date,
            },
            "raw_sha256": acquisition.raw_sha256,
            "cache_key": acquisition.cache_key,
            "cache_mode": acquisition.cache_mode,
            "acquired_at": acquisition.acquired_at,
            "quality_control": qc,
        }


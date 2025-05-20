from dataclasses import dataclass, field, asdict
import json
from datetime import datetime


@dataclass
class Experiment:
    model_name: str
    avg_AScore: float
    growth: float
    percent_clear_gen: float
    percent_cut_gen: float
    percent_wrong_gen: float

    model_config: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, path):
        # Преобразуем весь объект в словарь
        data = asdict(self)

        with open(path, "w", encoding='utf-8') as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
                default=str  # обработка несериализуемых объектов
            )

    @classmethod
    def from_json(cls, path):
        """Загрузка эксперимента из JSON файла"""
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)

        return cls(
            model_name=data["model_name"],
            avg_AScore=data["avg_AScore"],
            growth=data["growth"],
            percent_clear_gen=data["percent_clear_gen"],
            percent_cut_gen=data["percent_cut_gen"],
            percent_wrong_gen=data["percent_wrong_gen"],
            model_config=data["model_config"],
            timestamp=data["timestamp"]
        )
import random
from dataclasses import dataclass

@dataclass
class PollQuestion:
    text: str
    answer: str
    others: list[str]

    def options(self, shuffle: bool = True) -> tuple[list[str], int]:
        """
        Возвращает (список_опций, индекс_правильного_ответа).

        Если shuffle=True, опции перемешиваются, и индекс меняется.
        """
        opts = [self.answer] + list(self.others)
        correct_id = 0

        if shuffle:
            indices = list(range(len(opts)))
            random.shuffle(indices)
            opts = [opts[i] for i in indices]
            correct_id = indices.index(0)

        return opts, correct_id

POLL_QUESTIONS_RU: list[PollQuestion] = [
    PollQuestion(
        text="Какого цвета небо днём?",
        answer="Голубое",
        others=["Зелёное", "Красное", "Чёрное"],
    ),
    PollQuestion(
        text="Какой звук издаёт кошка?",
        answer="Мяу",
        others=["Гав", "Му", "Пи-пи"],
    ),
    PollQuestion(
        text="Какой звук издаёт собака?",
        answer="Гав",
        others=["Мяу", "Кря", "Му"],
    ),
    PollQuestion(
        text="Что делают пчёлы?",
        answer="Мёд",
        others=["Кетчуп", "Пластик", "Газ"],
    ),
    PollQuestion(
        text="Где живут рыбы?",
        answer="В воде",
        others=["В небе", "В шкафу", "В кастрюле по умолчанию"],
    ),
    PollQuestion(
        text="Сколько лап у кошки?",
        answer="4",
        others=["2", "3", "6"],
    ),
    PollQuestion(
        text="Сколько ног у человека?",
        answer="2",
        others=["4", "3", "6"],
    ),
    PollQuestion(
        text="Что надевают на ноги?",
        answer="Обувь",
        others=["Шапку", "Перчатки", "Очки"],
    ),
    PollQuestion(
        text="Что из этого обычно пьют?",
        answer="Воду",
        others=["Камень", "Подушку", "Ложку"],
    ),
    PollQuestion(
        text="Какое животное из списка умеет летать?",
        answer="Птица",
        others=["Собака", "Кошка", "Корова"],
    ),
    PollQuestion(
        text="Чем человек видит?",
        answer="Глазами",
        others=["Ушами", "Коленом", "Пальцем"],
    ),
    PollQuestion(
        text="Чем человек слышит?",
        answer="Ушами",
        others=["Ногами", "Глазами", "Пупком"],
    ),
    PollQuestion(
        text="Чем человек чувствует запахи?",
        answer="Носом",
        others=["Пяткой", "Ухом", "Локтем"],
    ),
    PollQuestion(
        text="Что надевают на голову зимой?",
        answer="Шапку",
        others=["Туфли", "Перчатки", "Носки"],
    ),
    PollQuestion(
        text="Что обычно несут куры?",
        answer="Яйца",
        others=["Камни", "Телефоны", "Автомобили"],
    ),
    PollQuestion(
        text="Что любит пить корова?",
        answer="Воду",
        others=["Пепси", "Бензин", "Краску"],
    ),
    PollQuestion(
        text="Как обычно передвигается змея?",
        answer="Ползёт",
        others=["Летает", "Скачет", "Бегает на лапах"],
    ),
    PollQuestion(
        text="Что делают дети на качелях?",
        answer="Качаются",
        others=["Моют посуду", "Чинят машину", "Спят"],
    ),
    PollQuestion(
        text="Где растут яблоки?",
        answer="На дереве",
        others=["Под водой", "В песке", "В холодильнике сами по себе"],
    ),
    PollQuestion(
        text="Куда человек надевает перчатки?",
        answer="На руки",
        others=["На ноги", "На голову", "На уши"],
    ),
]
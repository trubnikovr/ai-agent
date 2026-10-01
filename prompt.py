def build_explanation_prompt(
    topic: str,
        level: str,
        language: str = 'русский',
):
    explanation_prompt = f"""
    Ты преподаватель программирования.

Объясни тему: {topic}
Уровень ученика: {level}
Язык ответа: {language}

Требования:
- объясни простыми словами;
- приведи практический пример;
- не превышай 300 слов.
    """
    return explanation_prompt
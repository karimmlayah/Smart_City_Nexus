from django import template

register = template.Library()


@register.filter
def name_initials(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "?"
    parts = [p for p in text.split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        word = parts[0]
        return (word[:2] if len(word) >= 2 else word[:1]).upper()
    return (parts[0][:1] + parts[-1][:1]).upper()

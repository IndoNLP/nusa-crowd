def map_label(label) -> str:
    if label == -1:
        return "negative"
    elif label == 0:
        return "neutral"
    elif label == 1:
        return "positive"
    else:
        return label
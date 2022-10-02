import os
import pathlib
import re

root = pathlib.Path(__file__).parent.resolve()


def replace_writing(content, marker, chunk, inline=False):
    r = re.compile(
        r"<!\-\- {} starts \-\->.*<!\-\- {} ends \-\->".format(marker, marker),
        re.DOTALL,
    )
    if not inline:
        chunk = "\n{}\n".format(chunk)
    chunk = "<!-- {} starts -->{}<!-- {} ends -->".format(marker, chunk, marker)
    return r.sub(chunk, content)


def build_progress_bar(milestones={}):
    progress_bar = []
    progress_bar_template = "![Milestone {}](https://progress-bar.dev/{}/?title=Milestone%20{}%20({}%20Datasets%20Completed))"
    for index, target in enumerate(milestones):
        percentage = min(milestones[target] * 100 // target, 100)
        bar = progress_bar_template.format(index + 1, percentage, index + 1, target)
        progress_bar.append(bar)
    return progress_bar


def calculate_completed_dataset(base_dir="nusacrowd/nusa_datasets"):
    count = 0
    for path in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, path)) and path != "__pycache__":
            count += 1
    return count


if __name__ == "__main__":
    # read current readme
    readme_path = root / "README.md"
    readme = readme_path.open().read()

    readme_id_path = root / "README.id.md"
    readme_id = readme_id_path.open().read()

    # calculate progress
    target = [30, 60, 100, 150]
    count_completed_dataset = calculate_completed_dataset()
    milestones = {k: count_completed_dataset for k in target}
    progress_bar = build_progress_bar(milestones=milestones)
    entries_md = "\n\n".join(progress_bar)

    # Update entries
    rewritten_entries = replace_writing(readme, "milestone", entries_md)
    readme_path.open("w").write(rewritten_entries)

    rewritten_id_entries = replace_writing(readme_id, "milestone", entries_md)
    readme_id_path.open("w").write(rewritten_id_entries)

import re

def listToString(list_of_string) -> str:
    one_string = " ".join(list_of_string)
    one_string = re.sub(r" (-\/) ", r"\1", one_string)
    one_string = re.sub(r" ([:;?!%.,])", r"\1", one_string)
    one_string = re.sub(r"\( ([^)]+) \)", r"(\1)", one_string)
    one_string = re.sub(r"\' ([^\']+) \'", r"'\1'", one_string)
    one_string = re.sub(r"\" ([^\"]+) \"", r'"\1"', one_string)
    return one_string

def getAnswerString(list_of_string, answer_mask) -> str:
    answer_string = ""
    for i in range(len(answer_mask)):
        if (answer_mask[i] == 'B'):
            if(answer_string != ""):
                answer_string += ", "
            answer_string += list_of_string[i]
        elif (answer_mask[i] == 'I'):
            answer_string += " " + list_of_string[i]

    return answer_string

# passage1 = ['Jimmy', 'Arnold', '(', '54', ')', ',', 'pencipta', 'Kursi', 'Stroke', '(', 'Stroke', 'Muscle', 'Therapy', ')', ',', 'Rabu', '(', '24/8', ')', 'siang', ',', 'menyatakan', ',', 'seorang', 'karyawan', 'perusahaan', 'minyak', 'AS', 'di', 'Jakarta', 'yakni', 'Jary', 'Wayland', '(', '60', ')', 'sangat', 'tertarik', 'akan', 'kursi', 'buatannya', '.']
# answerm1 = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

# print(listToString(passage1))
# print()
# print(getAnswerString(passage1, answerm1))
import json
import django
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Graduation.settings")

django.setup()



def main():
    from ReadingTest.models import Reading, Q_A
    data_path = os.getcwd() + '\RACE\RACE\\test\\'
    all_path = os.walk(data_path)

    for path, dir_list, file_list in all_path:
        for file_name in file_list:
            cur_path = os.path.join(path, file_name)
            with open(cur_path, 'r') as f:
                data = json.load(f)
                answers = data['answers']
                options = data['options']
                questions = data['questions']
                article = data['article']
                id_ = data['id']
                Reading.objects.get_or_create(id=id_, article=article)
                r = Reading.objects.get(id=id_)
                for i in range(len(answers)):
                    answer = answers[i]
                    option = options[i]
                    question = questions[i]
                    Q_A.objects.get_or_create(reading=r, question=question, answer=answer,
                        option_a=option[0], option_b=option[1], 
                        option_c=option[2], option_d=option[3])

if __name__ == "__main__":
    main()
    print('Done!')

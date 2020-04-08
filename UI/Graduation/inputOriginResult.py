import json
import django
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Graduation.settings")

django.setup()


def main():
    from ReadingTest.models import Reading, Q_A
    data_path = os.getcwd() + '\predict_result_orgin\\'
    all_path = os.walk(data_path)

    for path, dir_list, file_list in all_path:
        for file_name in file_list:
            cur_path = os.path.join(path, file_name)
            with open(cur_path, 'r') as f:
                data = json.load(f)
                answer = data['answer']
                question = data['question']
                id_ = data['id']
                r = Reading.objects.get(id=id_)
                q = Q_A.objects.filter(reading=r, question=question)
                for ques in q:
                    ques.origin_prediction = answer
                    ques.save()


if __name__ == "__main__":
    main()
    print('Done!')

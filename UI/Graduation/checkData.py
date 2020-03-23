import json
import django
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Graduation.settings")

django.setup()


def main():
    from ReadingTest.models import Reading, Q_A
    reading_count = Reading.objects.count()
    qa_num = Q_A.objects.count()
    print("num of Reading: " + str(reading_count))
    print("num of Question: " + str(qa_num))

if __name__ == "__main__":
    main()
    print('Done!')

from django.db import models

# Create your models here.

class Reading(models.Model):
    id = models.CharField(max_length=20, primary_key=True, unique=True)#主键
    article = models.TextField()
    def __str__(self):
        return self.id


class Q_A(models.Model):
    reading = models.ForeignKey('Reading', on_delete=models.CASCADE)
    question = models.CharField(max_length=512)
    answer = models.CharField(max_length=2)
    option_a = models.CharField(max_length=512)
    option_b = models.CharField(max_length=512)
    option_c = models.CharField(max_length=512)
    option_d = models.CharField(max_length=512)

    def __str__(self):
        return self.question


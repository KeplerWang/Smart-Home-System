from django.db import models


class TempAndMoist(models.Model):
    s_time = models.CharField(max_length=32)
    s_temp = models.IntegerField(default= 26)
    s_moist = models.IntegerField(default= 50)


# 用户的类
class User(models.Model):
    id = models.AutoField('序号', primary_key=True)
    username = models.CharField('用户名', max_length=32)
    password = models.CharField('密码', max_length=32)
    email = models.EmailField('邮箱')
    mobile = models.CharField('手机', max_length=11)
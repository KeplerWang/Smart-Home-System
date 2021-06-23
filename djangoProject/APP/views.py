import datetime

from django.http import HttpResponse
from django.shortcuts import render, redirect
from pyecharts import options as opts
from pyecharts.charts import Line
from APP.models import TempAndMoist
from mysql import Sql
from pyecharts.faker import Faker
from pathlib import Path
import os

# from APP.newHtml import newchart
sql = Sql()


def hello(request):
    return HttpResponse("Hello world ! ")


def index(request):
    return render(request, 'index.html')


def base(request):
    tempAndmoist = TempAndMoist()
    status = sql.get_status()
    tempAndmoist.s_temp = status.get('Temperature')
    tempAndmoist.s_moist = status.get('Humidity')
    tempAndmoist.save()
    return render(request, 'base.html', context={
        "Temperature": tempAndmoist.s_temp,
        "Humidity": tempAndmoist.s_moist
    })


def face(request):
    return render(request, 'face.html')


def chart(request):
    temp, humidity, time = sql.get_recent()
    newchart(x=range(20), temp=temp, humidity=humidity)
    return render(request, 'chart.html')


def login(request):
    # 当前端点击登录按钮时，提交数据到后端，进入该POST方法
    if request.method == "POST":
        # 获取用户名和密码
        username = request.POST.get("username")
        passwd = request.POST.get("password")
        # 在前端传回时也将跳转链接传回来
        next_url = request.POST.get("next_url")

        # 对用户进行验证，假设正确的用户名密码为"aaa", "123"
        if username == 'hust' and passwd == 'hust1037':
            # 判断用户一开始是不是从login页面进入的
            # 如果跳转链接不为空并且跳转页面不是登出页面，则登录成功后跳转，否则直接进入主页
            if next_url and next_url != "/logout/":
                response = redirect(next_url)
            else:
                response = redirect("/base/")
            return response
        # 若用户名或密码失败,则将提示语与跳转链接继续传递到前端
        else:
            error_msg = "用户名或密码不正确，请重新尝试"
            return render(request, "login.html", {
                'login_error_msg': error_msg,
                'next_url': next_url,
            })
    # 若没有进入post方法，则说明是用户刚进入到登录页面。用户访问链接形如下面这样：
    # http://host:port/login/?next=/next_url/
    # 拿到跳转链接
    next_url = request.GET.get("next", '')

    # 直接将跳转链接也传递到后端
    return render(request, "login.html", {
        'next_url': next_url,
    })


def look(request):
    # 显示出页面需要的参数值
    press_button = list(request.POST.keys())
    status = sql.get_status()
    if len(press_button) > 0:
        click = list(request.POST.keys())[0]
        if click == 'Wetter_On':
            status['Wetter'] = 1
            status['ctrl_wetter'] = 1
        elif click == 'Wetter_Off':
            status['Wetter'] = 0
            status['ctrl_wetter'] = 1
        elif click == 'Wetter_None':
            status['ctrl_wetter'] = 0
        elif click == 'Door_On':
            status['Door'] = 1
            status['ctrl_door'] = 1
        elif click == 'Door_Off':
            status['Door'] = 0
            status['ctrl_door'] = 1
        elif click == 'Door_None':
            status['ctrl_door'] = 0
        elif click == 'Fan_On':
            status['Fan'] = 1
            status['ctrl_fan'] = 1
        elif click == 'Fan_Off':
            status['Fan'] = 0
            status['ctrl_fan'] = 1
        elif click == 'Fan_None':
            status['ctrl_fan'] = 0
        elif click == 'Curtain_Up':
            status['Curtain'] = 1
            status['ctrl_curtain'] = 1
        elif click == 'Curtain_Down':
            status['Curtain'] = 0
            status['ctrl_curtain'] = 1
        elif click == 'Curtain_None':
            status['ctrl_curtain'] = 0
        status['Time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql.update_status(status)
    status = {key: '开' if value else '关' for key, value in status.items()}
    return render(request, 'look.html', context=status)


def deleteTempAndMoist(request):
    # 删除特定时间的温湿度
    tempandmoist = TempAndMoist.objects.get(pk=2)
    tempandmoist.delete()
    return HttpResponse("已将删除")


def newchart(x, temp, humidity):
    Line1 = (
        Line()
            .add_xaxis(xaxis_data=x)
            .add_yaxis("湿度", humidity)
            .extend_axis(
            yaxis=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value} °C"), interval=5
            )
        )
            .set_series_opts(label_opts=opts.LabelOpts(is_show=True))
            .set_global_opts(
            title_opts=opts.TitleOpts(title="温湿度曲线"),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value} %")),
        )
    )
    line = Line().add_xaxis(xaxis_data=x).add_yaxis("温度", temp, yaxis_index=1)
    Line1.overlap(line)
    Line1.render(os.path.join(Path(__file__).resolve().parent.parent, 'templates/chart.html'))

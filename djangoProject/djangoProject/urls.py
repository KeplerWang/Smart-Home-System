from django.conf.urls import url
from django.contrib import admin
from APP import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^logout', views.login, name='off'),
    url(r'^base/', views.base, name='base'),
    url(r'^login/', views.login),  # 登录动作
    url(r'^index/', views.index, name='index'),
    url(r'^look/', views.look, name='look'),
    url(r'^face/', views.face, name='face'),
    url(r'^chart/', views.chart, name='chart'),
    url('^$', views.login)
]

import pymysql


class Sql:
    _message = ''

    def _template_fun(self, sql):
        database, cur = self._creat_connection()
        cur.execute(sql)
        database.commit()
        self._message = cur.fetchall()
        self._close_connection(database, cur)

    @staticmethod
    def _creat_connection():
        try:
            database = pymysql.connect(host='localhost', user='root', password='123456',
                                       port=3306, db='test')
            cur = database.cursor()
        except pymysql.err.OperationalError:
            database = pymysql.connect(host='localhost', user='root', password='123456',
                                       port=3306)
            cur = database.cursor()
            cur.execute('create database test default character set utf8;')
            database.commit()
            cur.execute('use test;')
            sql = """CREATE TABLE device_status (
                id int auto_increment primary key,
                door tinyint(1),
                fan tinyint(1),
                wetter tinyint(1),
                curtain tinyint(1),
                ctrl_curtain tinyint(1) DEFAULT 0,
                ctrl_door tinyint(1) DEFAULT 0,
                ctrl_wetter tinyint(1) DEFAULT 0,
                ctrl_fan tinyint(1) DEFAULT 0,
                time datetime
                );"""
            cur.execute(sql)
            sql = """CREATE TABLE status (
                id int auto_increment primary key,
                temperature float,
                humidity float,
                time datetime
                );"""
            cur.execute(sql)
        return database, cur

    @staticmethod
    def _close_connection(database, cur):
        cur.close()
        database.close()

    def insert_device_status(self, status):
        sql = """INSERT INTO device_status(door, fan, wetter, 
                curtain, ctrl_wetter, ctrl_door, ctrl_curtain, ctrl_fan, time) 
                VALUES(%d, %d, %d, %d, %d, %d, %d, %d, '%s')""" % (status.get('Door'), status.get('Fan'),
                                                                   status.get('Wetter'), status.get('Curtain'),
                                                                   status.get('ctrl_wetter'), status.get('ctrl_door'),
                                                                   status.get('ctrl_curtain'), status.get('ctrl_fan'),
                                                                   status.get('Time'))
        self._template_fun(sql)

    def insert_status(self, status):
        temperature, humidity, time = status.get('Temperature'), status.get('Humidity'), status.get('Time')
        sql = """INSERT INTO status(temperature, humidity, time)
                        VALUES(%.1f, %3.1f, '%s');""" % (temperature, humidity, time)
        self._template_fun(sql)

    def get_status(self):
        sql = """SELECT * FROM device_status ORDER BY id DESC;"""
        self._template_fun(sql)
        try:
            door, fan, wetter, curtain, ctrl_curtain, ctrl_door, ctrl_wetter, ctrl_fan = self._message[0][1:-1]
        except:
            door, fan, wetter, curtain, ctrl_curtain, ctrl_door, ctrl_wetter, ctrl_fan = [0] * 8

        sql = """SELECT * FROM status ORDER BY id DESC;"""
        self._template_fun(sql)
        try:
            temperature, humidity = self._message[0][1:-1]
        except:
            temperature, humidity = 0, 0
        return {
            'Temperature': temperature,
            'Humidity': humidity,
            'Door': door,
            'Fan': fan,
            'Wetter': wetter,
            'Curtain': curtain,
            'ctrl_curtain': ctrl_curtain,
            'ctrl_door': ctrl_door,
            'ctrl_wetter': ctrl_wetter,
            'ctrl_fan': ctrl_fan
        }

    def update_status(self, status):
        sql = """SELECT MAX(id) FROM device_status;"""
        self._template_fun(sql)
        max_id = self._message[0][0]
        sql = f"""UPDATE device_status SET door = {status.get('Door')},
                                           fan = {status.get('Fan')},
                                           wetter = {status.get('Wetter')},
                                           curtain = {status.get('Curtain')},
                                           ctrl_curtain = {status.get('ctrl_curtain')},
                                           ctrl_door = {status.get('ctrl_door')},
                                           ctrl_wetter = {status.get('ctrl_wetter')},
                                           ctrl_fan = {status.get('ctrl_fan')},
                                           time = '{status.get('Time')}'
                                           WHERE id = {max_id};"""
        self._template_fun(sql)

    def get_recent(self):
        sql = """SELECT temperature, humidity, time FROM status ORDER BY id desc LIMIT 20;"""
        self._template_fun(sql)
        print(self._message)
        return list(reversed([i[0] for i in self._message])), \
               list(reversed([i[1] for i in self._message])), \
               list(reversed([i[2].strftime("%m-%d %H:%M") for i in self._message]))


if __name__ == '__main__':
    s = Sql()
    print(s.get_status())
    status = {
        'Temperature': 23.1,
        'Humidity': 70,
        'Door': 0,
        'Fan': 0,
        'Wetter': 0,
        'Curtain': 0,
        'ctrl_curtain': 0,
        'ctrl_door': 0,
        'ctrl_wetter': 0,
        'ctrl_fan': 0,
        'Time': '2021-06-21 23:26:50'
    }
    s.insert_status(status)
    print(s.get_status())

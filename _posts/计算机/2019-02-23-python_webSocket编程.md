---
layout: post
title: python-WebSocket编程
category: 计算机
tags: 计算机
keywords: python webSocketIO
description: 
---

经常使用HTTP协议来进行网络编程，最近需要服务器主动向客户端发起通信请求，传统的HTTP协议显得捉襟见肘---AJAX轮询来和服务端交互实在太丑太low。于是着手研究WebSocket协议（是的，以前都是简单GET,POST了事😓）。

可惜这方面python的入门资料不是很多，选择无非就是[Django](https://www.fullstackpython.com/django.html)，[Flask](http://flask.pocoo.org/) 和[tornado](http://www.tornadoweb.org/en/stable/websocket.html) 等几个。以前使用Flask较多，为能减小学习成本，也尽可能兼容以前HTTP协议的应用，我选择了Flask-SocketIO。可惜这方面文档不是很多，于是就翻译了下面这篇文章：

**以下全文翻译自**[**博客**](https://blog.miguelgrinberg.com/post/easy-websockets-with-flask-and-gevent)

这个周末我决定忙中偷闲，给自己放个小假，先不写[这本书](http://flaskbook.com/)了。花时间好好做一个想了很久的项目。这个项目是一个全新的Flask扩展，我觉得很酷。 

很高兴介绍[Flask-SocketIO](https://github.com/miguelgrinberg/Flask-SocketIO),简单易用的WebSocket通信在Flask上的扩展。 

## 什么是WebSocket？ 

[WebSocket](http://en.wikipedia.org/wiki/WebSocket)是HTML5中引入的新的通信协议。主要被网络客户端与服务端实现，所以也可以在web外使用。 

不同于HTTP通信，WebSocket通信是长久，双向的服务端和客户端的通信通道，也就是任何一端都可以初始化数据交换。一旦建立连接，连接一直保持建立直到一方断开。 

对于需要实时交换信息的低延迟网络站点或游戏应用，WebSocket连接显得非常有用。这个协议诞生之前还有很多不是很高效的方法也能取得同样的效果，比如[Comet](http://en.wikipedia.org/wiki/Comet_(programming))。 

下面的浏览器支持WebSocket协议： 

\* Chrome 14 

\* Safari 6 

\* Firefox 6 

\* Internet Explorer 10 

## 什么是SocketIO？ 

[SocketIO](http://socket.io/)是一个跨浏览器JavaScript库，它抽象了真正的交换协议中的客户端应用。对于现代浏览器，SocketIO使用WebSocket协议，但是对于不支持WebSocket的老旧的浏览器，SocketIO使用其他对两端都何时的较老的解决方案来仿真模拟连接。 

重要的是所有的应用实例中它都提供相同的接口。不同的连接机制被抽象成通用API，所以使用SocketIO你可以相当确定任何浏览器可以连接到你的应用，而且是最高效适合这个浏览器的方法。 

## Flask-Sockets呢? 

前一段时间Kenneth Reitz发布了Flask-Sockets---另一个Flask扩展，这可以让Flask应用使用WebSocket协议。 

Flask-Sockets和Flask-SocketIO之间最大的区别是前者包装了原始WebSocket协议（使用了gevent-websocket项目），所以它尽可以用在支持WebSocket的现代浏览器上。而Flask-SocketIO却在老式浏览器上直接降级。 

另一个最大的不同是Flask-SocketIO实现了JS库SocketIO暴露的消息传输协议。而Flask-Sockets仅实现了通信通道---在上传输的数据完全依据应用来定。 

Flask-SocketIO也创建了事件处理环境，that is close to that of regular view functions, including the creation of application and request contexts.There are some important exceptions to this explained in the documentation, however. 

## Flask-SocketIO服务端 

安装Flask-Sockets很容易： 

```
$ pip install flask-socketio 
```

注意Flask-SocketIO依赖gevent库，目前它仅可以在python2上运行（译者注：python3.6测试也可以）。gevent很快也会对python3支持。 

下面是一个Flask-SocketIO在Flask应用上的实现例子： 

```
from flask import Flask, render_template 
from flask_socketio import SocketIO, emit 

app = Flask(name) 
app.config['SECRET_KEY'] = 'secret!' 
socketio = SocketIO(app) 

@app.route('/') 
def index(): 
    return render_template('index.html') 

@socketio.on('my event', namespace='/test') 
def test_message(message): 
    emit('my response', {'data': message['data']}) 

@socketio.on('my broadcast event', namespace='/test') 
def test_message(message): 
    emit('my response', {'data': message['data']}, broadcast=True) 

@socketio.on('connect', namespace='/test') 
def test_connect(): 
    emit('my response', {'data': 'Connected'}) 

@socketio.on('disconnect', namespace='/test') 
def test_disconnect(): 
    print('Client disconnected') 

if name == 'main': 
    socketio.run(app) 
```

这个扩展使用常用方法来初始化。扩展提供了 **run()** 方法来简化服务的启动。 

这个方法启动了gevent，仅在web 服务中支持。使用gunicorn的gevent应该依然可以工作。**run()** 方法接受可选参数**host**和**port**，但是默认状态下它会监听**localhost:5000**,就像传统的Flask开发web应用服务一样。 

这个应用中仅有传统的路由是 **/**,提供**index.html**---这个例子的客户端实现。 

使用**socketio.on**装饰器来接收从客户端发送来的WebSocket信息。**socketio.on**的第一个参数是event名称。**connect**, **disconnect**, **message**和**json**是SocketIO产生的特殊events。其它event名称被认为是定制events。 

**connect**,**disconnect**不言自明，**message**传递字符串，**JSON**传递JSON。 

在**socketio.on**装饰器上添加可选参数**namespace**可以在命名空间内定义events。Namespace允许客户端在一个socket中(多路复用)打开多个和服务端的连接。当未指定namespace时，events被附着在默认的全局namespace上。 

Flask服务端可以用Flask-SocketIO提供的**send()** 和 **emit()** 发送events。**send()** 发送标准字符串或JSON信息给客户端，**emit()** 在定制event名称下发送信息。默认信息被发送到连接的客户端上，但当可选参数**broadcast**设置为True的时候，所有namespace下连接的客户端都会收到信息。 

## SocketIO 客户端 

准备好试试Javascript? 例子服务器中使用的index.html包含一些使用jQuery和SocketIO的客户端应用。相关代码如下: 

```
$(document).ready(function(){ 
     var socket = io.connect('http://' + document.domain + ':' + location.port + '/test'); 
         socket.on('my response', function(msg) { 
             $('#log').append('<p>Received: ' + msg.data + '</p>'); 
    }); 
    
    $('form#emit').submit(function(event) { 
        socket.emit('my event', {data: $('#emit_data').val()}); 
        return false; 
    }); 

    $('form#broadcast').submit(function(event) { 
        socket.emit('my broadcast event', {data:$('#broadcast_data').val()}); 
        return false; 
    }); 

});  
```

**socket**变量在SocketIO连接到服务器中初始化。注意 **\/test** namespace是如何在连接URL中指定的。如果不使用namespace来连接，直接使用不带任何参数的 **io.connect()** 就已经够了。 

**socket.on()** 用在客户端一侧来定义事件处理(注释：event handler)。在这个例子中定制的**my response** event通过在message中增加id名为**log**的元素来被处理(发送？)，其中**log**附着了**data**属性。这些元素在页面的HTML部分被定义。 

接下来的两部分重载了两个表单的提交按钮事件，这就替代了原本在HTTP上提交表单的回调函数（默认触发的）。 

表单中id为**emit**的提交事件发送给服务器一个名为**my event**的消息函数，这个函数的数据是JSON，JSON的data属性是表单中文本框内的值。 

id为**broadcast**的第二个表单也做了同样的事情。不同的是在称为**my broadcast event**的event中发送数据。 

再看一眼服务端代码，可以详细看到这两个定制event的处理代码。对于**my event**时间，服务端仅简单地用**my response** event回应给客户端数据。对于**my broadcast event**也差不多做同样的事，但不同的是它广播给所有连接的客户端。 

你可以在[github repo](https://github.com/miguelgrinberg/Flask-SocketIO/blob/master/example/templates/index.html)中看到整个HTML文档。 

## 运行例子 

你需要先从github下载整个代码库来运行。你有两个选择来下载： 

\* 使用git [clone the repo](https://github.com/miguelgrinberg/Flask-SocketIO) 

\* [作为zip](https://github.com/miguelgrinberg/Flask-SocketIO/archive/master.zip)直接下载 

例子应用就在**example**目录内，直接**cd**就能到。 

为保持系统的python解释器干净(译者：看自己选择吧)，可以创建虚拟环境来工作。 

```
$ virtualenv venv 
$ . venv/bin/activate 
```

之后你需要安装依赖: 

```
(venv) $ pip install -r requirements.txt 
```

最后直接运行： 

```
(venv) $ python app.py  
```

现在打开浏览器，直接打开[**http://localhost:5000**](http://localhost:5000/)你就会得到如下所示的页面： 



![img](https://pic1.zhimg.com/v2-df1ffddc85698020b185558535635634_b.png)

上面两个文本框中的文本将会通过SocketIO连接发送给服务端，服务端会直接再把它返回给客户端，客户端会把信息添加在页面的"Receive"部分。在"Receive"部分可以看到服务端通过**connect**事件发送的信息。 

如果你给这个应用连接第二个浏览器会更有趣。我这个例子中使用Firefox和Chrome两个来测试了，实际上你机器上的任何两个浏览器都可以。如果你想通过不同的机器来访问服务端，也可以达到同样的效果---你需要改变**socketio.run(app, host='0.0.0.0')** 来来让服务端监听局域网内接口。 

当你打开两个或更多客户端的时候，在左侧表格提交的文本的客户端才会收到服务端的回声响应。如果在右侧的表格提交，服务端广播的回声信息会发送给所有连接的客户端。 

如果客户端断开连接（比如你关掉了浏览器页面），服务端在几秒后会检测到，并会发送一个disconnect 事件给应用。终端会打印出这个信息。 

## 最后 

本扩展的更多解释去哪个参阅[文档](http://flask-socketio.readthedocs.org/en/latest/)。如果有意的话可以fork后提交代码。 

希望你能使用这个扩展做出很酷的应用。我就用它做了很多有意思的扩展。 

如果你做了些有趣的应用，请直接在留言区贴出链接。 

**更新：** Flask-SocketIO 0.3增加了room。这可以让链接对象不必使用广播选项来发送信息给每一个人。 

**更新2：** Flask-SocketIO 1.0 增加了Python3.3的支持，而且提供了服务端选择gevent或eventlet的选项，如同标准Flask服务。 

**全文翻译自**[**博客**](https://blog.miguelgrinberg.com/post/easy-websockets-with-flask-and-gevent)


---
title: go连接starrocks执行超时
date: 2025-01-24
categories: [文章, '202501']
tags: [问题解决, golang]
---

## 背景
同事问连接go-sql-driver/mysql连接starrocks需要特殊配置吗, 正常连接后执行sql每次都会超时

![报错信息](/commons/202501/1.png)

## 排查

没用过starrocks, 看了starrocks的文档, 看到starrocks支持mysql协议, 所以用go-sql-driver/mysql连接starrocks
测了下mysql-client 直连starrocks, 连接没问题, 执行也没问题, 看来mysql协议是支持的

golang go-sql-driver/mysql连接也都正常, 但是执行sql就每次都会超时 那就找其他详细报错信息

![报错信息](/commons/202501/2.png)

看起来是 io timeout,  确认了眼读写超时配置 3s,  难道服务器是美东 库是国内 网络不稳定?
查了下库也是美东, ping了下网络也没有问题, 而且发现执行sql是必现超时, 不像网络的影响
![ping](/commons/202501/3.png)


接着在服务器抓了下包, 发现包有点问题, 
golang ->  starrocks 发送 sql
starrocks -> golang 返回执行结果
上面这部分都很快 接着下面就开始异常了
3s后  golang端主动发包 然后关闭连接

![问题](/commons/202501/4.png)


光看包也看不明白...按上面抛出报错信息的mysql/connection.go:185开始追代码看

```shell

	// Read Result
	columnCount, err := stmt.readPrepareResultPacket()
	if err == nil {
		if stmt.paramCount > 0 {
			if err = mc.readUntilEOF(); err != nil {
				return nil, err
			}
		}

		if columnCount > 0 {
			err = mc.readUntilEOF()
		}
	}

	return stmt, err
}

```

看起来是starrocks没有发回eof,  golang就一直等, 直到3s读超时 重置连接
诶 不应该阿...
看这个方法是prepare 突然灵光一现 把sql改了, 不用?的语法,同时用 Raw() 执行 成功了....
猜测是 starrocks的版本是不是不支持prepare 或者两边版本不一致协议没对上 
然后就搜到下面这个issue 看来是个bug 那.... 


[issue](https://github.com/StarRocks/starrocks/issues/52688)

## 解决
让她试试暂时改用Raw()..

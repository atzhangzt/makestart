from sqlalchemy import create_engine, NUMERIC,Table,Column,Integer,String,MetaData,ForeignKey,REAL,Sequence
import tushare as ts
import pandas as pd


def createTable(code):
    tableName = 'stock_' + code
    stocks_table = Table(tableName,metadata,
        Column('id',Integer,Sequence(tableName + '_id'),primary_key=True),
        Column('code', String(6), nullable=True),
        Column('date',  String(10), nullable=True),
        Column('open', REAL, nullable=True),
        Column('high', REAL, nullable=True),
        Column('close', REAL, nullable=True),
        Column('low',REAL, nullable=True),
        Column('volume', REAL, nullable=True),
        Column('amount ', REAL, nullable=True)
        )
    metadata.create_all()

engine = create_engine('sqlite:///data.sqlite')  
metadata = MetaData(engine)

fid =0
result = ts.get_stock_basics()

for code in result.index:
    fid =0
    createTable(code)
    dr = result.loc[code].values[0:-1]
    startTime = str(dr[14])
    year = startTime[0:4]
    
    tableName = 'stock_' + code
    stocks_tb = Table(tableName, metadata, autoload=True)
    ins = stocks_tb.insert()
    conn = engine.connect()

    iyear = int(year)
    while iyear < 2019:
        start = str(iyear) +'-01-01'
        end = str(iyear+1)+'-01-01'
        iyear = iyear + 1
        hist_d = ts.get_h_data(code,start=start,end=end)

        for date in hist_d.index:
            fid = fid +1
            dr = hist_d.loc[date].values[0:-1]
            print(dr)s
            conn.execute(ins,id =fid, code=code,date=date,open=dr[0],high=dr[1],close=dr[2],low=dr[3],volume=dr[4],amount=dr[5])
        
        

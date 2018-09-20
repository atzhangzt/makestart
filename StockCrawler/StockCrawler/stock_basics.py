from sqlalchemy import create_engine, NUMERIC,Table,Column,Integer,String,MetaData,ForeignKey,REAL,Sequence
import tushare as ts
import pandas as pd

engine = create_engine('sqlite:///data.sqlite')  
metadata = MetaData(engine)

stocks_table = Table('stock_basics',metadata,
        Column('code', String(6), primary_key=True),
        Column('name',  String(30), nullable=True),
        Column('industry', String(30), nullable=True),
        Column('area', String(30), nullable=True),
        Column('pe', REAL, nullable=True),
        Column('outstanding',REAL, nullable=True),
        Column('totals', REAL, nullable=True),
        Column('totalAssets', REAL, nullable=True),
        Column('liquidAssets', REAL, nullable=True),
        Column('fixedAssets', REAL, nullable=True),
        Column('reserved', REAL, nullable=True),
        Column('reservedPerShare', REAL, nullable=True),
        Column('eps', REAL, nullable=True),
        Column('bvps', REAL, nullable=True),
        Column('pb', REAL, nullable=True),
        Column('timeToMarket', NUMERIC, nullable=True)
        )
metadata.create_all()

stocks_tb = Table('stock_basics', metadata, autoload=True)
ins = stocks_tb.insert()
conn = engine.connect()

result = ts.get_stock_basics()
for index in result.index:
    dr = result.loc[index].values[0:-1]
    value = conn.execute(ins,code=index,name=dr[0],industry=dr[1],area=dr[2],pe=dr[3],outstanding=dr[4],totals=dr[5],totalAssets=dr[6],liquidAssets=dr[7],fixedAssets=dr[8],reserved=dr[9],reservedPerShare=dr[10],eps=dr[11],bvps=dr[12],pb=dr[13],timeToMarket=dr[14])
    print(index)

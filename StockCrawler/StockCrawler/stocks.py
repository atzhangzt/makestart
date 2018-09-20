from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table,Column,Integer,String,MetaData,ForeignKey,Float

Base=declarative_base()   #基类

class stock(Base):
    __tablename__='stocks'   #表名 
    code=Column(Integer,primary_key=True)
    name=Column(String)
    industry=Column(String)
    area=Column(String)
    pe=Column(Float)
    outstanding=Column(Float)
    totals=Column(Float)
    totalAssets=Column(Float)
    def getTableName():
        return 'stocks'



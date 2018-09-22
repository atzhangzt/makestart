from sqlalchemy import create_engine, NUMERIC,Table,Column,Integer,String,MetaData,ForeignKey,REAL,Sequence
from sqlalchemy.orm import sessionmaker
import tushare as ts

def createTable(code):
    tableName = 'stock_' + code
    stocks_table = Table(tableName,metadata,
        Column('id',Integer,Sequence(tableName + '_id'),primary_key=True),
        Column('code', String(6), nullable=True),
        Column('ktype',  String(2), nullable=True),
        Column('date',  String(10), nullable=True),
        Column('open', REAL, nullable=True),
        Column('high', REAL, nullable=True),
        Column('close', REAL, nullable=True),
        Column('low',REAL, nullable=True),
        Column('volume', REAL, nullable=True)
        )
    metadata.create_all()


def get_k_data(ktype,session,code,start,end ,fid):
    hd = ts.get_k_data(code,start=start,end=end)
    range1 = hd.shape[0]
    for x in range(range1):
        fid = fid+1
        row = hd.ix[x]
        date = row[0]
        open = row[1]
        close = row[2]
        high = row[3]      
        low = row[4]
        volume = row[5]            
        session.execute(ins,{'id' :fid,'ktype':ktype, 'code':code,'date':date,'open':open,'high':high,'close':close,'low':low,'volume':volume})
    return fid
 
engine = create_engine('sqlite:///data.sqlite')  
metadata = MetaData(engine)
Session = sessionmaker(bind=engine)


result = ts.get_stock_basics()
for code in result.index:
    createTable(code)
    dr = result.loc[code].values[0:-1]
    startTime = str(dr[14])
    year = startTime[0:4]
    
    tableName = 'stock_' + code
    stocks_tb = Table(tableName, metadata, autoload=True)
    ins = stocks_tb.insert()
    fid =0
    iyear = int(year)
    session = Session()
    while iyear < 2019:

        start = str(iyear) +'-01-01'
        end = str(iyear+1)+'-01-01'
        iyear = iyear + 1
        print(tableName +'-d-'+ str(iyear))
        fid = get_k_data('M',session,code,start,end,fid)
        fid = get_k_data('W',session,code,start,end,fid)
        fid = get_k_data('D',session,code,start,end,fid)
        fid = get_k_data('60',session,code,start,end,fid)
        fid = get_k_data('30',session,code,start,end,fid)
        fid = get_k_data('15',session,code,start,end,fid)
        fid = get_k_data('5',session,code,start,end,fid)
    session.commit()
    session.close()


        
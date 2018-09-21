from sqlalchemy import create_engine, NUMERIC,Table,Column,Integer,String,MetaData,ForeignKey,REAL,Sequence,Date
import tushare as ts

def createTable(code):
    tableName = 'stock_' + code
    stocks_table = Table(tableName,metadata,
        Column('id',Integer,Sequence(tableName + '_id'),primary_key=True),
        Column('code', String(6), nullable=True),
        Column('date',  Date, nullable=True),
        Column('open', REAL, nullable=True),
        Column('high', REAL, nullable=True),
        Column('close', REAL, nullable=True),
        Column('low',REAL, nullable=True),
        Column('volume', REAL, nullable=True),
        Column('amount', REAL, nullable=True)
        )
    metadata.create_all()

engine = create_engine('sqlite:///data.sqlite')  
metadata = MetaData(engine)

fid =0
result = ts.get_stock_basics()

for m in range(20000):
    k = ts.get_h_data('002337', start='2015-01-01', end='2015-02-16',pause=5)
    print(m)

for code in result.index:
    createTable(code)
    dr = result.loc[code].values[0:-1]
    startTime = str(dr[14])
    year = startTime[0:4]
    
    tableName = 'stock_' + code
    stocks_tb = Table(tableName, metadata, autoload=True)
    ins = stocks_tb.insert()
    conn = engine.connect()

    fid =0
    iyear = int(year)
    while iyear < 2019:
        start = str(iyear) +'-01-01'
        end = str(iyear+1)+'-01-01'
        iyear = iyear + 1
        print(iyear)
        hist_d = ts.get_h_data(code,start=start,end=end,retry_count=20000000000,pause=0.1)       
        
        range1 = hist_d.shape[0]        
        for x in range(range1):
            fid = fid+1
            row = hist_d.ix[x]           
            open = row['open']
            high = row['high']
            close = row['close']
            low = row['low']
            volume = row['volume']            
            amount = row['amount']
            date = row.name
            conn.execute(ins,id =fid, code=code,date=date,open=open,high=high,close=close,low=low,volume=volume,amount=amount)
            
        


from sqlalchemy import create_engine, NUMERIC,Table,Column,Integer,String,MetaData,ForeignKey,REAL,Sequence
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

engine = create_engine('sqlite:///data.sqlite')  
metadata = MetaData(engine)

fid =0
result = ts.get_stock_basics()
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

        print(tableName +'-d-'+ str(iyear))
        hist_d = ts.get_k_data(code,start=start,end=end)        
        range1 = hist_d.shape[0]
        for x in range(range1):
            fid = fid+1
            row = hist_d.ix[x]        
            date = row[0]
            open = row[1]
            close = row[2]
            high = row[3]           
            low = row[4]
            volume = row[5]
            conn.execute(ins,id =fid,ktype='D', code=code,date=date,open=open,high=high,close=close,low=low,volume=volume)
        
        print(tableName +'-w-'+ str(iyear))
        hist_W = ts.get_k_data(code,start=start,end=end,ktype='W')        
        range1 = hist_W.shape[0]
        for x in range(range1):
            fid = fid+1
            row = hist_W.ix[x]        
            date = row[0]
            open = row[1]
            close = row[2]
            high = row[3]           
            low = row[4]
            volume = row[5]
            conn.execute(ins,id =fid,ktype='W', code=code,date=date,open=open,high=high,close=close,low=low,volume=volume)
             
        print(tableName +'-m-'+ str(iyear))
        hist_M = ts.get_k_data(code,start=start,end=end,ktype='M')        
        range1 = hist_M.shape[0]
        for x in range(range1):
            fid = fid+1
            row = hist_M.ix[x]        
            date = row[0]
            open = row[1]
            close = row[2]
            high = row[3]           
            low = row[4]
            volume = row[5]
            conn.execute(ins,id =fid,ktype='M', code=code,date=date,open=open,high=high,close=close,low=low,volume=volume)
              
        print(tableName +'-5-'+ str(iyear))
        hist_5 = ts.get_k_data(code,start=start,end=end,ktype='5')        
        range1 = hist_5.shape[0]
        for x in range(range1):
            fid = fid+1
            row = hist_5.ix[x]        
            date = row[0]
            open = row[1]
            close = row[2]
            high = row[3]           
            low = row[4]
            volume = row[5]
            conn.execute(ins,id =fid,ktype='5', code=code,date=date,open=open,high=high,close=close,low=low,volume=volume)
        
        print(tableName +'-15-'+ str(iyear))
        hist_15 = ts.get_k_data(code,start=start,end=end,ktype='15')        
        range1 = hist_15.shape[0]
        for x in range(range1):
            fid = fid+1
            row = hist_15.ix[x]        
            date = row[0]
            open = row[1]
            close = row[2]
            high = row[3]           
            low = row[4]
            volume = row[5]
            conn.execute(ins,id =fid,ktype='15', code=code,date=date,open=open,high=high,close=close,low=low,volume=volume)
        
        print(tableName +'-30-'+ str(iyear))
        hist_30 = ts.get_k_data(code,start=start,end=end,ktype='30')        
        range1 = hist_30.shape[0]
        for x in range(range1):
            fid = fid+1
            row = hist_30.ix[x]        
            date = row[0]
            open = row[1]
            close = row[2]
            high = row[3]           
            low = row[4]
            volume = row[5]
            conn.execute(ins,id =fid,ktype='30', code=code,date=date,open=open,high=high,close=close,low=low,volume=volume)

        print(tableName +'-60-'+ str(iyear))
        hist_60 = ts.get_k_data(code,start=start,end=end,ktype='60')        
        range1 = hist_60.shape[0]
        for x in range(range1):
            fid = fid+1
            row = hist_60.ix[x]        
            date = row[0]
            open = row[1]
            close = row[2]
            high = row[3]           
            low = row[4]
            volume = row[5]
            conn.execute(ins,id =fid,ktype='60', code=code,date=date,open=open,high=high,close=close,low=low,volume=volume)


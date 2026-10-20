import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
C='#2F5D8A'; A='#E07A2F'
OUT='/home/claude/out/'; R={}
raw=pd.read_csv('/mnt/user-data/uploads/dataset.csv')
d=raw.copy()
# ---------- cleaning ----------
d['created_at']=pd.to_datetime(d.created_at); d['actual_delivery_time']=pd.to_datetime(d.actual_delivery_time)
d['delivery_min']=(d.actual_delivery_time-d.created_at).dt.total_seconds()/60
d['hour']=d.created_at.dt.hour; d['dow']=d.created_at.dt.day_name()
d['market']=d.market_id.apply(lambda x:'Unknown' if pd.isna(x) else f'Market {int(x)}')
d['category']=d.store_primary_category.fillna('unknown')
top=d.category.value_counts().head(12).index
d['category_grp']=np.where(d.category.isin(top),d.category,'other')
# data-quality flags
for c in ['total_onshift_partners','total_busy_partners','total_outstanding_orders']:
    d.loc[d[c]<0,c]=np.nan            # impossible negative counts
d['busy_ratio']=(d.total_busy_partners/d.total_onshift_partners.replace(0,np.nan)).clip(upper=1.5)
d['orders_per_partner']=d.total_outstanding_orders/d.total_onshift_partners.replace(0,np.nan)
R['raw_rows']=len(raw)
R['null_delivery']=int(raw.actual_delivery_time.isna().sum())
R['gt180']=int((d.delivery_min>180).sum())
R['max_min']=float(d.delivery_min.max())
v=d[d.delivery_min.between(1,180)].copy()          # valid delivery-time set
R['valid_rows']=len(v)
R['mean_all_raw']=float(d.delivery_min.mean()); R['mean_valid']=float(v.delivery_min.mean()); R['median_valid']=float(v.delivery_min.median())
dow_order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
# Q1
q1=d.market.value_counts().rename_axis('market').reset_index(name='orders'); q1['pct']=q1.orders/q1.orders.sum()*100
q1=q1.sort_values('market'); R['q1']=q1.round(1).to_dict('records')
fig,ax=plt.subplots(figsize=(6,3.4)); ax.bar(q1.market,q1.orders,color=C); ax.set_title('Orders by market'); ax.set_ylabel('Orders')
plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(OUT+'charts/q1.png'); plt.close()
# Q2
q2=v.groupby('category').delivery_min.agg(['mean','count']).query('count>=500').sort_values('mean')
R['q2_fast']=q2.head(3).round(1).reset_index().to_dict('records'); R['q2_slow']=q2.tail(5).round(1).reset_index().to_dict('records')
R['q2_n']=len(q2)
sel=pd.concat([q2.head(8),q2.tail(8)]).drop_duplicates()
fig,ax=plt.subplots(figsize=(6.2,4.4)); ax.barh(sel.index,sel['mean'],color=C); ax.axvline(v.delivery_min.mean(),color=A,ls='--',label='Overall mean')
ax.set_xlabel('Avg delivery time (min)'); ax.set_title('Fastest and slowest categories (≥500 orders)'); ax.legend(); plt.tight_layout(); plt.savefig(OUT+'charts/q2.png'); plt.close()
# Q3
q3=d.hour.value_counts().sort_index(); R['q3_top']=q3.sort_values(ascending=False).head(5).to_dict()
fig,ax=plt.subplots(figsize=(6.4,3.4)); ax.bar(q3.index,q3.values,color=C); ax.set_xlabel('Hour of day (timestamp as recorded, UTC)'); ax.set_ylabel('Orders'); ax.set_title('Orders by hour placed'); plt.tight_layout(); plt.savefig(OUT+'charts/q3.png'); plt.close()
# Q4
q4=d.dow.value_counts().reindex(dow_order); R['q4']=q4.to_dict()
fig,ax=plt.subplots(figsize=(6,3.4)); ax.bar([x[:3] for x in dow_order],q4.values,color=C); ax.set_title('Orders by day of week'); ax.set_ylabel('Orders'); plt.tight_layout(); plt.savefig(OUT+'charts/q4.png'); plt.close()
# Q5
clean5=d[(d.total_items<=50)&(d.subtotal>0)]
R['q5_pearson']=float(d.total_items.corr(d.subtotal)); R['q5_pearson_clean']=float(clean5.total_items.corr(clean5.subtotal)); R['q5_spearman']=float(d.total_items.corr(d.subtotal,method='spearman'))
R['q5_slope']=float(np.polyfit(clean5.total_items,clean5.subtotal,1)[0])/100
s=clean5.sample(15000,random_state=1)
fig,ax=plt.subplots(figsize=(6,3.6)); ax.scatter(s.total_items,s.subtotal/100,s=5,alpha=.25,color=C); ax.set_xlabel('Items in order'); ax.set_ylabel('Subtotal'); ax.set_title('Items vs subtotal (sample, ≤50 items)'); plt.tight_layout(); plt.savefig(OUT+'charts/q5.png'); plt.close()
# Q6
dm=d.delivery_min.dropna()
q1_,q3_=v.delivery_min.quantile([.25,.75]); iqr=q3_-q1_; hi=q3_+1.5*iqr
R['q6']=dict(p50=float(v.delivery_min.quantile(.5)),p90=float(v.delivery_min.quantile(.9)),p95=float(v.delivery_min.quantile(.95)),p99=float(v.delivery_min.quantile(.99)),skew=float(v.delivery_min.skew()),std=float(v.delivery_min.std()),iqr_hi=float(hi),n_iqr=int((v.delivery_min>hi).sum()),pct_iqr=float((v.delivery_min>hi).mean()*100),n_fast=int((v.delivery_min<15).sum()))
fig,ax=plt.subplots(1,2,figsize=(8,3.4)); ax[0].hist(v.delivery_min,bins=80,color=C); ax[0].axvline(hi,color=A,ls='--',label=f'IQR fence {hi:.0f} min'); ax[0].set_title('Delivery time distribution'); ax[0].set_xlabel('Minutes'); ax[0].legend()
ax[1].boxplot(v.delivery_min,vert=False,showfliers=True,flierprops=dict(markersize=2,alpha=.3)); ax[1].set_title('Box plot'); ax[1].set_xlabel('Minutes'); plt.tight_layout(); plt.savefig(OUT+'charts/q6.png'); plt.close()
# Q7
R['q7']=float((v.delivery_min<=30).mean()*100); R['q7_45']=float((v.delivery_min<=45).mean()*100); R['q7_60']=float((v.delivery_min<=60).mean()*100)
q7m=v.groupby('market').delivery_min.apply(lambda s:(s<=30).mean()*100).round(1); R['q7_market']=q7m.to_dict()
# Q8
g=d.dropna(subset=['total_busy_partners']).copy()
R['q8_corr']=float(g.total_busy_partners.corr(g.total_outstanding_orders))
hrs=g.groupby(g.created_at.dt.floor('h')).agg(orders=('market','size'),busy=('total_busy_partners','mean'),onshift=('total_onshift_partners','mean'))
R['q8_hourly_corr']=float(hrs.orders.corr(hrs.busy)); R['q8_order_corr']=float(g.total_busy_partners.corr(g.total_onshift_partners))
g['bq']=pd.qcut(g.total_busy_partners,10,duplicates='drop')
fig,ax=plt.subplots(figsize=(6,3.6)); ax.scatter(hrs.busy,hrs.orders,s=6,alpha=.4,color=C); ax.set_xlabel('Avg busy partners (hourly)'); ax.set_ylabel('Orders placed in that hour'); ax.set_title('Order volume vs busy partners'); plt.tight_layout(); plt.savefig(OUT+'charts/q8.png'); plt.close()
# Q9
q9=v.groupby('dow').delivery_min.agg(['mean','median']).reindex(dow_order); R['q9']=q9.round(1).to_dict('index')
fig,ax=plt.subplots(figsize=(6,3.4)); ax.bar([x[:3] for x in dow_order],q9['mean'],color=C); ax.set_ylim(q9['mean'].min()-5,q9['mean'].max()+2); ax.set_ylabel('Avg delivery (min)'); ax.set_title('Avg delivery time by day of week'); plt.tight_layout(); plt.savefig(OUT+'charts/q9.png'); plt.close()
# extra: hour effect
qh=v.groupby('hour').delivery_min.agg(['mean','count']); R['hour_mean']=qh['mean'].round(1).to_dict()
# Q10
st=v.groupby('store_id').agg(avg=('delivery_min','mean'),n=('delivery_min','size'),category=('category','first'),market=('market',lambda s:s.mode().iat[0]))
R['stores_total']=len(st); R['stores_ge20']=int((st.n>=20).sum())
t5=st[st.n>=20].sort_values('avg',ascending=False).head(5).round(1).reset_index(); R['q10_top5']=t5.to_dict('records')
t5_any=st.sort_values('avg',ascending=False).head(5).round(1).reset_index(); R['q10_top5_any']=t5_any.to_dict('records')
sd_m=v.groupby('market').delivery_min.agg(['mean','std']).round(1); R['q10_sd_market']=sd_m.to_dict('index')
sd_c=v.groupby('category').delivery_min.agg(['mean','std','count']).query('count>=500').sort_values('std',ascending=False).round(1); R['q10_sd_cat']=sd_c.head(5).reset_index().to_dict('records'); R['q10_sd_cat_low']=sd_c.tail(3).reset_index().to_dict('records')
fig,ax=plt.subplots(figsize=(6.4,3.2)); ax.barh(t5.store_id.str[:8]+' ('+t5.category+')',t5.avg,color=C); ax.invert_yaxis(); ax.set_xlabel('Avg delivery (min)'); ax.set_title('Top 5 slowest stores (≥20 orders)'); plt.tight_layout(); plt.savefig(OUT+'charts/q10.png'); plt.close()
# Q11
m=v.dropna(subset=['busy_ratio']).copy()
q11=m.groupby('market').agg(delivery=('delivery_min','mean'),busy_ratio=('busy_ratio','mean'),onshift=('total_onshift_partners','mean'),outstanding=('total_outstanding_orders','mean'),opp=('orders_per_partner','mean'),n=('delivery_min','size')).round(2)
R['q11']=q11.to_dict('index')
R['q11_corr_busy']=float(m.busy_ratio.corr(m.delivery_min)); R['q11_corr_onshift']=float(m.total_onshift_partners.corr(m.delivery_min)); R['q11_corr_opp']=float(m.orders_per_partner.corr(m.delivery_min)); R['q11_corr_out']=float(m.total_outstanding_orders.corr(m.delivery_min))
R['q11_corr_by_market']={k:round(float(x.busy_ratio.corr(x.delivery_min)),3) for k,x in m.groupby('market') if len(x)>200}
m['bb']=pd.cut(m.busy_ratio,[0,.5,.7,.85,.95,1.01,2],labels=['<50%','50-70%','70-85%','85-95%','95-100%','>100%'])
bb=m.groupby('bb',observed=True).delivery_min.agg(['mean','count']).round(1); R['q11_bands']=bb.to_dict('index')
fig,ax=plt.subplots(figsize=(6,3.4)); ax.bar(bb.index.astype(str),bb['mean'],color=C); ax.set_xlabel('Share of on-shift partners that are busy'); ax.set_ylabel('Avg delivery (min)'); ax.set_ylim(bb['mean'].min()-4,bb['mean'].max()+2); ax.set_title('Delivery time vs partner utilisation'); plt.tight_layout(); plt.savefig(OUT+'charts/q11.png'); plt.close()
# Q12 - no customer id
R['customer_col']=[c for c in raw.columns if 'cust' in c.lower()]
# proxy: store order frequency tiers (new/low-volume vs established)
sc=d.groupby('store_id').size(); d['store_orders']=d.store_id.map(sc)
d['store_tier']=pd.cut(d.store_orders,[0,10,50,1e9],labels=['Low (≤10 orders)','Mid (11-50)','High (>50)'])
tier=d.groupby('store_tier',observed=True).agg(orders=('subtotal','size'),items=('total_items','mean'),subtotal=('subtotal','mean')).round(1); tier['subtotal']=(tier.subtotal/100).round(2); R['tier']=tier.to_dict('index')
# protocol & other
pr=v.groupby('order_protocol').delivery_min.agg(['mean','count']).round(1); R['protocol']=pr.to_dict('index')
# regression drivers (market 6 excluded, needs partner data)
import numpy.linalg as la
fc=['total_items','subtotal','num_distinct_items','total_onshift_partners','total_busy_partners','total_outstanding_orders']
mm=m.dropna(subset=fc); X=mm[fc].copy(); y=mm.delivery_min
Z=(X-X.mean())/X.std(); Zc=np.c_[np.ones(len(Z)),Z.values]; beta=la.lstsq(Zc,y.values,rcond=None)[0]
R['reg_beta_std']=dict(zip(X.columns,beta[1:].round(2))); pred=Zc@beta; R['reg_r2']=float(1-((y-pred)**2).sum()/((y-y.mean())**2).sum())
# hour x delivery & busy ratio by hour
fig,ax=plt.subplots(figsize=(6.4,3.4)); ax.plot(qh.index,qh['mean'],marker='o',color=C); ax.set_xlabel('Hour placed (UTC)'); ax.set_ylabel('Avg delivery (min)'); ax.set_title('Avg delivery time by hour placed'); plt.tight_layout(); plt.savefig(OUT+'charts/hour_delivery.png'); plt.close()
# market delivery chart
mk=v.groupby('market').delivery_min.agg(['mean',lambda s:(s<=30).mean()*100]); mk.columns=['mean','on30']; R['market_summary']=mk.round(1).to_dict('index')
fig,ax=plt.subplots(figsize=(6,3.4)); ax.bar(mk.index,mk['mean'],color=C); ax.set_ylabel('Avg delivery (min)'); ax.set_title('Avg delivery time by market'); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(OUT+'charts/market_delivery.png'); plt.close()
# data quality
R['dq']=dict(null_market=int(raw.market_id.isna().sum()),null_cat=int(raw.store_primary_category.isna().sum()),null_proto=int(raw.order_protocol.isna().sum()),null_partners=int(raw.total_onshift_partners.isna().sum()),neg_partner=int((raw.total_onshift_partners<0).sum()+(raw.total_busy_partners<0).sum()+(raw.total_outstanding_orders<0).sum()),neg_minprice=int((raw.min_item_price<0).sum()),zero_sub=int((raw.subtotal==0).sum()),big_items=int((raw.total_items>50).sum()),min_gt_max=int((raw.min_item_price>raw.max_item_price).sum()),n_cat=int(raw.store_primary_category.nunique()),n_store=int(raw.store_id.nunique()),m6_missing=float(raw[raw.market_id==6].total_onshift_partners.isna().mean()*100))
R['date_range']=[str(d.created_at.min()),str(d.created_at.max())]
# save cleaned
cols=['market','created_at','actual_delivery_time','delivery_min','hour','dow','store_id','category','category_grp','order_protocol','total_items','subtotal','num_distinct_items','min_item_price','max_item_price','total_onshift_partners','total_busy_partners','total_outstanding_orders','busy_ratio']
d[cols].to_csv('/mnt/user-data/outputs/porter_cleaned.csv',index=False)
json.dump(R,open(OUT+'results.json','w'),indent=1,default=str)
print(json.dumps(R,indent=1,default=str))

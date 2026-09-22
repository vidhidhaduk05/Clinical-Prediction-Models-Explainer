export const mean = a => a.reduce((s,x)=>s+x,0)/a.length;
export const sum = a => a.reduce((s,x)=>s+x,0);
export const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
export const sigmoid = x => 1/(1+Math.exp(-x));
export const logit = p => Math.log(clamp(p,1e-9,1-1e-9)/(1-clamp(p,1e-9,1-1e-9)));
export function rng(seed=42){let a=seed>>>0;return()=>{a+=0x6D2B79F5;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
export function shuffle(a,random){const b=[...a];for(let i=b.length-1;i>0;i--){let j=Math.floor(random()*(i+1));[b[i],b[j]]=[b[j],b[i]];}return b;}
export function quantile(a,p){const b=[...a].sort((x,y)=>x-y);const z=(b.length-1)*p,i=Math.floor(z);return b[i]+(b[Math.min(i+1,b.length-1)]-b[i])*(z-i);}
export function metrics(y,p){const e=p.map((v,i)=>v-y[i]),ss=sum(y.map(v=>(v-mean(y))**2)),mse=mean(e.map(v=>v*v));return{mae:mean(e.map(Math.abs)),rmse:Math.sqrt(mse),signed:mean(e),r2:ss>0?1-sum(e.map(v=>v*v))/ss:NaN};}
export function ols(x,y){const mx=mean(x),my=mean(y);const den=sum(x.map(v=>(v-mx)**2));const b=den?sum(x.map((v,i)=>(v-mx)*(y[i]-my)))/den:0;return{a:my-b*mx,b};}
export function predict(model,x){return x.map(v=>model.a+model.b*v);}
export function solve(A,b){const m=A.map((row,i)=>[...row,b[i]]),n=b.length;for(let i=0;i<n;i++){let q=i;for(let j=i+1;j<n;j++)if(Math.abs(m[j][i])>Math.abs(m[q][i]))q=j;[m[i],m[q]]=[m[q],m[i]];if(Math.abs(m[i][i])<1e-12)return Array(n).fill(0);const v=m[i][i];for(let k=i;k<=n;k++)m[i][k]/=v;for(let j=0;j<n;j++)if(j!==i){const c=m[j][i];for(let k=i;k<=n;k++)m[j][k]-=c*m[i][k];}}return m.map(row=>row[n]);}
export function fitBasis(X,y,penalty=1e-8){const k=X[0].length,A=Array.from({length:k},()=>Array(k).fill(0)),b=Array(k).fill(0);X.forEach((row,i)=>{for(let j=0;j<k;j++){b[j]+=row[j]*y[i];for(let l=0;l<k;l++)A[j][l]+=row[j]*row[l];}});for(let j=1;j<k;j++)A[j][j]+=penalty;return solve(A,b);}
export function rcs(x,knots=[1,4,7,9]){const k=knots.length,last=knots[k-1],prev=knots[k-2],cube=v=>Math.max(0,v)**3;return [1,x,...knots.slice(0,-2).map(t=>(cube(x-t)-cube(x-prev)*(last-t)/(last-prev)+cube(x-last)*(prev-t)/(last-prev))/(last-knots[0])**2)];}
export const dot=(a,b)=>sum(a.map((v,i)=>v*b[i]));
export function auc(y,p){const yes=p.filter((_,i)=>y[i]===1),no=p.filter((_,i)=>y[i]===0);if(!yes.length||!no.length)return NaN;return sum(yes.map(a=>sum(no.map(b=>a>b?1:a===b?.5:0))))/(yes.length*no.length);}
export function confusion(y,p,t){let tp=0,fp=0,tn=0,fn=0;y.forEach((v,i)=>{if(p[i]>=t){v?tp++:fp++;}else{v?fn++:tn++;}});return{tp,fp,tn,fn,sensitivity:tp/(tp+fn),specificity:tn/(tn+fp),ppv:tp/(tp+fp)};}
export const brier=(y,p)=>mean(p.map((v,i)=>(v-y[i])**2));
export function netBenefit(y,p,t){const c=confusion(y,p,t);return c.tp/y.length-c.fp/y.length*t/(1-t);}
export function bh(ps,q=.05){const order=ps.map((p,i)=>({p,i})).sort((a,b)=>a.p-b.p),m=ps.length;let cutoff=-1;order.forEach((x,j)=>{if(x.p<=(j+1)/m*q)cutoff=j;});return order.map((x,j)=>({...x,rank:j+1,threshold:(j+1)/m*q,reject:j<=cutoff}));}
export function rubin(estimates,variances){const m=estimates.length,W=mean(variances),Q=mean(estimates),B=sum(estimates.map(v=>(v-Q)**2))/(m-1),T=W+(1+1/m)*B;return{Q,W,B,T,se:Math.sqrt(T)};}
export function repeatedCV(x,y,k=5,repeats=20,seed=42){const random=rng(seed),all=Array(x.length).fill(0),repMetrics=[];for(let r=0;r<repeats;r++){const idx=shuffle(x.map((_,i)=>i),random),p=Array(x.length);for(let f=0;f<k;f++){const test=idx.filter((_,j)=>j%k===f),train=idx.filter((_,j)=>j%k!==f),fit=ols(train.map(i=>x[i]),train.map(i=>y[i]));test.forEach(i=>p[i]=fit.a+fit.b*x[i]);}p.forEach((v,i)=>all[i]+=v/repeats);repMetrics.push(metrics(y,p));}return{predictions:all,pooled:metrics(y,all),repetitionMean:mean(repMetrics.map(m=>m.r2)),repMetrics};}
export function bootstrapIndices(n,random){const drawn=Array.from({length:n},()=>Math.floor(random()*n)),seen=new Set(drawn);return{drawn,oob:Array.from({length:n},(_,i)=>i).filter(i=>!seen.has(i))};}
export function km(time,event){const unique=[...new Set(time.filter((_,i)=>event[i]))].sort((a,b)=>a-b);let s=1;const points=[[0,1]];unique.forEach(t=>{const risk=time.filter(x=>x>=t).length,d=time.filter((x,i)=>x===t&&event[i]).length;points.push([t,s]);s*=1-d/risk;points.push([t,s]);});points.push([Math.max(...time),s]);return points;}
export const toyX=[2,3,5,6,8,9,11,13,15,18,21,24];
export const toyY=[8,20,14,28,25,45,38,56,48,83,66,108];
export const binaryY=[0,0,1,0,0,1,0,1,0,1,1,1];
export const binaryP=[.08,.12,.19,.23,.3,.38,.43,.56,.63,.71,.82,.91];

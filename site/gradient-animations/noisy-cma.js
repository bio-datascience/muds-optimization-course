// Full covariance CMA-ES in two dimensions; noisy values affect selection only.
window.makeNoisyCMA=function({landscape,start,noise,batch,seed}){
const c=Math.cos(.65),s=Math.sin(.65),R=landscape==='rastrigin',f=x=>R?20+x.reduce((a,v)=>a+v*v-10*Math.cos(2*Math.PI*v),0):(c*x[0]+s*x[1])**2+35*(-s*x[0]+c*x[1])**2;
let state=seed>>>0;const rand=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return(state+.5)/4294967296},normal=()=>Math.sqrt(-2*Math.log(rand()))*Math.cos(2*Math.PI*rand());
const p=2,q=12,mu=6,w=Array.from({length:mu},(_,i)=>Math.log(mu+.5)-Math.log(i+1));let sw=w.reduce((a,b)=>a+b);w.forEach((a,i)=>w[i]/=sw);const me=1/w.reduce((a,b)=>a+b*b,0),cc=(4+me/p)/(p+4+2*me/p),cs=(me+2)/(p+me+5),c1=2/((p+1.3)**2+me),cm=Math.min(1-c1,2*(me-2+1/me)/((p+2)**2+me)),damp=1+2*Math.max(0,Math.sqrt((me-1)/(p+1))-1)+cs,chi=Math.sqrt(p)*(1-1/(4*p)+1/(21*p*p));
let m=R?(start==='perturbed'?[2.6,2.3]:[2.5,2.5]):[-3.5,2.5],C=[[1,0],[0,1]],pc=[0,0],ps=[0,0],sigma=1.2,best=f(m),evals=0,frames=[];
function eig(){let theta=.5*Math.atan2(2*C[0][1],C[0][0]-C[1][1]),co=Math.cos(theta),si=Math.sin(theta),disc=Math.hypot(C[0][0]-C[1][1],2*C[0][1]),d1=Math.sqrt(Math.max(1e-12,(C[0][0]+C[1][1]+disc)/2)),d2=Math.sqrt(Math.max(1e-12,(C[0][0]+C[1][1]-disc)/2));return {co,si,d1,d2};}
for(let t=0;t<=250;t++){
 const e=eig(),population=[];
 for(let r=0;r<q;r++){let z=[normal(),normal()],y=[e.co*e.d1*z[0]-e.si*e.d2*z[1],e.si*e.d1*z[0]+e.co*e.d2*z[1]],x=m.map((a,j)=>a+sigma*y[j]),truth=f(x),obs=truth;for(let b=0;b<batch;b++)obs+=noise*normal()/batch;evals+=batch;population.push({x,y,truth,observed:obs});best=Math.min(best,truth);}
 population.sort((a,b)=>a.observed-b.observed);let next=[0,0];for(let i=0;i<mu;i++)for(let j=0;j<p;j++)next[j]+=w[i]*population[i].x[j];const y=next.map((a,j)=>(a-m[j])/sigma),rot=[e.co*y[0]+e.si*y[1],-e.si*y[0]+e.co*y[1]],white=[e.co*rot[0]/e.d1-e.si*rot[1]/e.d2,e.si*rot[0]/e.d1+e.co*rot[1]/e.d2];
 const inverse=[[e.co**2/e.d1**2+e.si**2/e.d2**2,e.co*e.si*(1/e.d1**2-1/e.d2**2)],[e.co*e.si*(1/e.d1**2-1/e.d2**2),e.si**2/e.d1**2+e.co**2/e.d2**2]].map(row=>row.map(v=>2*v/(sigma*sigma)));
 frames.push({t,x:[...m],f:f(m),best,gradient:[0,0],gradient_norm:0,sample_gradient:null,eta:0,shift:0,model:inverse,population:population.map((a,i)=>({...a,selected:i<mu})),sigma,q,f_evals:evals,gradient_evals:0,next:t<250?next:null});
 ps=ps.map((a,j)=>(1-cs)*a+Math.sqrt(cs*(2-cs)*me)*white[j]);let norm=Math.hypot(...ps),hs=norm/Math.sqrt(1-(1-cs)**(2*(t+1)))<(1.4+2/(p+1))*chi?1:0;pc=pc.map((a,j)=>(1-cc)*a+hs*Math.sqrt(cc*(2-cc)*me)*y[j]);
 C=C.map((row,i)=>row.map((v,j)=>(1-c1-cm)*v+c1*(pc[i]*pc[j]+(1-hs)*cc*(2-cc)*v)+cm*population.slice(0,mu).reduce((a,r,k)=>a+w[k]*r.y[i]*r.y[j],0)));sigma=Math.max(1e-7,Math.min(5,sigma*Math.exp(cs/damp*(norm/chi-1))));m=next;
}
return{method:'cma',landscape,noise,batch,seed,decay:0,frames,generated:true,oracle:'function',parameters:{q}};
};

/* Shared finite-sum gradient oracle: four linear perturbations average to zero. */
window.makeNoisyRun=function({method,landscape,start,noise,batch,seed,decay,rate,schedule='constant'}){
const c=Math.cos(.65),s=Math.sin(.65),isR=landscape==='rastrigin',L=isR?2+40*Math.PI**2:70;
const f=x=>isR?20+x.reduce((a,v)=>a+v*v-10*Math.cos(2*Math.PI*v),0):(c*x[0]+s*x[1])**2+35*(-s*x[0]+c*x[1])**2;
const grad=x=>isR?x.map(v=>2*v+20*Math.PI*Math.sin(2*Math.PI*v)):[2*c*(c*x[0]+s*x[1])-70*s*(-s*x[0]+c*x[1]),2*s*(c*x[0]+s*x[1])+70*c*(-s*x[0]+c*x[1])];
const H=x=>isR?[[2+40*Math.PI**2*Math.cos(2*Math.PI*x[0]),0],[0,2+40*Math.PI**2*Math.cos(2*Math.PI*x[1])]]:[[2*c*c+70*s*s,-68*c*s],[-68*c*s,2*s*s+70*c*c]];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1],mv=(A,x)=>A.map(r=>dot(r,x)),inv=A=>{let d=A[0][0]*A[1][1]-A[0][1]*A[1][0];return [[A[1][1]/d,-A[0][1]/d],[-A[1][0]/d,A[0][0]/d]]};
let state=seed>>>0;const rand=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return state/4294967296};
let x=isR?(start==='perturbed'?[2.6,2.3]:[2.5,2.5]):[-3.5,2.5],m=[0,0],v=[0,0],A=[[1,0],[0,1]],best=Infinity,fe=0,ge=0,frames=[];
for(let t=0;t<=450;t++){
 const g=grad(x),fx=f(x);ge++;fe++;best=Math.min(best,fx);let z=[0,0],ids=[];
 for(let j=0;j<batch;j++){let k=Math.floor(4*rand());ids.push(k);z[k>>1]+=(k%2?-1:1)*noise*Math.SQRT2/batch;}
 const gh=g.map((a,j)=>a+z[j]);let eta=(['gd','sgd'].includes(method)?rate:1)/L;if(method==='sgd'&&schedule==='cooling')eta*=t<100?1:1/(1+(t-100)/25)**1.5;let d=gh.map(a=>-a),shift=0,model=null,adaptive=null,update=null;
 if(method==='adam'||method==='adamw'){
  eta=.06;for(let j=0;j<2;j++){m[j]=.9*m[j]+.1*gh[j];v[j]=.999*v[j]+.001*gh[j]**2;}
  adaptive=m.map((a,j)=>-eta*(a/(1-.9**(t+1)))/(Math.sqrt(v[j]/(1-.999**(t+1)))+1e-8));
 }else if(method==='newton'||method==='bfgs'){
  if(method==='newton'){model=H(x);const a=model[0][0],b=model[0][1],d0=model[1][1],low=(a+d0-Math.hypot(a-d0,2*b))/2;shift=Math.max(0,1-low);model=[[a+shift,b],[b,d0+shift]];d=mv(inv(model),gh).map(a=>-a);}else{d=mv(A,gh).map(a=>-a);model=inv(A);}
  if(dot(d,gh)>=0)d=gh.map(a=>-a);
  // Freeze the sampled linear perturbation throughout this iteration's line search.
  eta=1;for(let j=0;j<30;j++){const y=x.map((a,k)=>a+eta*d[k]);fe++;if(f(y)+dot(z,y)<=fx+dot(z,x)+1e-4*eta*dot(gh,d))break;eta*=.5;}
 }
 let next=adaptive?x.map((a,j)=>a+adaptive[j]-(method==='adamw'?eta*decay*a:0)):x.map((a,j)=>a+eta*d[j]);
 const frame={t,x:[...x],f:fx,best,gradient:g,gradient_norm:Math.hypot(...g),sample_gradient:gh,sample_ids:ids,eta,shift,model,adaptive_step:adaptive||[0,0],f_evals:fe,gradient_evals:ge,next:t<450?next:null};
 if(method==='bfgs'){
  // Reuse the current batch at both endpoints for a consistent secant pair.
  const y=grad(next).map((a,j)=>a-g[j]),ss=next.map((a,j)=>a-x[j]),ys=dot(y,ss);ge++;
  if(ys>1e-10*Math.hypot(...y)*Math.hypot(...ss)){const ay=mv(A,y),coef=(ys+dot(y,ay))/(ys*ys);A=A.map((row,i)=>row.map((a,j)=>a+coef*ss[i]*ss[j]-(ay[i]*ss[j]+ss[i]*ay[j])/ys));update='Same-batch secant update';}else update='Curvature update skipped';frame.bfgs_update=update;
 }
 frames.push(frame);if(!next.every(Number.isFinite))break;x=next;
}
return {method,landscape,seed,noise,batch,decay,frames,generated:true,parameters:{L,initial_rate:method.startsWith('adam')?.06:1/L}};
};

precision highp float;

uniform sampler2D textureSampler;
uniform sampler2D depthSampler;
uniform sampler2D normalSampler;
uniform sampler2D prevColorSampler;

varying vec2 vUV;
uniform vec2 oneTexel;
uniform float Time;

float random(vec2 st) {
  return fract(sin(dot(st.xy, vec2(12.9898,78.233)))* 43758.5453123)*2.-1.;
}
struct Track {
  vec3 c0;
  float d0;
  vec3 c1;
  float d1;
  vec3 c2;
  float d2;
  vec3 c3;
  float d3;
  vec2 c;
};
void track(
  inout float w,
  inout vec2 p,
  Track center,
  Track target
) {
  const float cw = .5;
  const float dw = .0;
  const float th = 1./16.;
  float c = 0.;
  float d = 0.;
  const float abc = 1.;
  const vec3 ctr = vec3(abc,abc,abc);
  c += abs(dot(center.c0-target.c0,ctr));
  d += abs(center.d0-target.d0);
  c += abs(dot(center.c1-target.c1,ctr));
  d += abs(center.d1-target.d1);
  c += abs(dot(center.c2-target.c2,ctr));
  d += abs(center.d2-target.d2);
  c += abs(dot(center.c3-target.c3,ctr));
  d += abs(center.d3-target.d3);
  c *= cw;
  d *= dw;
  float v = float(c + d < th);
  w += v;
  p += v * target.c / (th - c - d);
}

void main() {    

    const float noiseness = .4;
    const float ts = 8.;
    vec4 tex = texture(textureSampler, vUV);
    const float zScale = 1.1;
    float depth = pow(zScale,texture(depthSampler, vUV).r*16.)/16.;

    // vec2[36] mks = vec2[](
    //   vec2(-5.,-5.),
    //   vec2(-5.,-3.),
    //   vec2(-5.,-1.),
    //   vec2(-5.,1.),
    //   vec2(-5.,3.),
    //   vec2(-5.,5.),
    //   vec2(-3.,-5.),
    //   vec2(-3.,-3.),
    //   vec2(-3.,-1.),
    //   vec2(-3.,1.),
    //   vec2(-3.,3.),
    //   vec2(-3.,5.),
    //   vec2(-1.,-5.),
    //   vec2(-1.,-3.),
    //   vec2(-1.,-1.),
    //   vec2(-1.,1.),
    //   vec2(-1.,3.),
    //   vec2(-1.,5.),
    //   vec2(1.,-5.),
    //   vec2(1.,-3.),
    //   vec2(1.,-1.),
    //   vec2(1.,1.),
    //   vec2(1.,3.),
    //   vec2(1.,5.),
    //   vec2(3.,-5.),
    //   vec2(3.,-3.),
    //   vec2(3.,-1.),
    //   vec2(3.,1.),
    //   vec2(3.,3.),
    //   vec2(3.,5.),
    //   vec2(5.,-5.),
    //   vec2(5.,-3.),
    //   vec2(5.,-1.),
    //   vec2(5.,1.),
    //   vec2(5.,3.),
    //   vec2(5.,5.)
    // );
    // vec3[36] pcls;
    // float[36] pdps;
    // for(int i = 0; i < 36; i++){
    //   mks[i] *= oneTexel;
    //   mks[i] *= ts;
    //   mks[i] += vUV;
    //   vec2 cr = mks[i];
    //   pcls[i] = texture(prevColorSampler, cr).rgb;
    //   // pdps[i] = dot(texture(DiffuseCDepthSampler, cr),vec4(1./(255.*255.*255.),1./(255.*255.),1./255.,1.));
    // }
    // vec3 mk1c = texture(textureSampler, mks[14]).rgb;
    // vec3 mk2c = texture(textureSampler, mks[15]).rgb;
    // vec3 mk3c = texture(textureSampler, mks[20]).rgb;
    // vec3 mk4c = texture(textureSampler, mks[21]).rgb;
    // float mk1d = texture(depthSampler, mks[14]).r;
    // float mk2d = texture(depthSampler, mks[15]).r;
    // float mk3d = texture(depthSampler, mks[20]).r;
    // float mk4d = texture(depthSampler, mks[21]).r;
    // Track center = Track(mk1c,mk1d,mk2c,mk2d,mk3c,mk3d,mk4c,mk4d,vec2(0.,0.));

    // Track[25] trs = Track[](
    //   Track(pcls[0],pdps[0],pcls[1],pdps[1],pcls[6],pdps[6],pcls[7],pdps[7],vec2(-4.,-4.)),
    //   Track(pcls[1],pdps[1],pcls[2],pdps[2],pcls[7],pdps[7],pcls[8],pdps[8],vec2(-2.,-4.)),
    //   Track(pcls[2],pdps[2],pcls[3],pdps[3],pcls[8],pdps[8],pcls[9],pdps[9],vec2(0.,-4.)),
    //   Track(pcls[3],pdps[3],pcls[4],pdps[4],pcls[9],pdps[9],pcls[10],pdps[10],vec2(2.,-4.)),
    //   Track(pcls[4],pdps[4],pcls[5],pdps[5],pcls[10],pdps[10],pcls[11],pdps[11],vec2(4.,-4.)),

    //   Track(pcls[6],pdps[6],pcls[7],pdps[7],pcls[12],pdps[12],pcls[13],pdps[13],vec2(-4.,-2.)),
    //   Track(pcls[7],pdps[7],pcls[8],pdps[8],pcls[13],pdps[13],pcls[14],pdps[14],vec2(-2.,-2.)),
    //   Track(pcls[8],pdps[8],pcls[9],pdps[9],pcls[14],pdps[14],pcls[15],pdps[15],vec2(0.,-2.)),
    //   Track(pcls[9],pdps[9],pcls[10],pdps[10],pcls[15],pdps[15],pcls[16],pdps[16],vec2(2.,-2.)),
    //   Track(pcls[10],pdps[10],pcls[11],pdps[11],pcls[16],pdps[16],pcls[17],pdps[17],vec2(4.,-2.)),

    //   Track(pcls[12],pdps[12],pcls[13],pdps[13],pcls[18],pdps[18],pcls[19],pdps[19],vec2(-4.,0.)),
    //   Track(pcls[13],pdps[13],pcls[14],pdps[14],pcls[19],pdps[19],pcls[20],pdps[20],vec2(-2.,0.)),
    //   Track(pcls[14],pdps[14],pcls[15],pdps[15],pcls[20],pdps[20],pcls[21],pdps[21],vec2(0.,0.)),
    //   Track(pcls[15],pdps[15],pcls[16],pdps[16],pcls[21],pdps[21],pcls[22],pdps[22],vec2(2.,0.)),
    //   Track(pcls[16],pdps[16],pcls[17],pdps[17],pcls[22],pdps[22],pcls[23],pdps[23],vec2(4.,0.)),

    //   Track(pcls[18],pdps[18],pcls[19],pdps[19],pcls[24],pdps[24],pcls[25],pdps[25],vec2(-4.,2.)),
    //   Track(pcls[19],pdps[19],pcls[20],pdps[20],pcls[25],pdps[25],pcls[26],pdps[26],vec2(-2.,2.)),
    //   Track(pcls[20],pdps[20],pcls[21],pdps[21],pcls[26],pdps[26],pcls[27],pdps[27],vec2(0.,2.)),
    //   Track(pcls[21],pdps[21],pcls[22],pdps[22],pcls[27],pdps[27],pcls[28],pdps[28],vec2(2.,2.)),
    //   Track(pcls[22],pdps[22],pcls[23],pdps[23],pcls[28],pdps[28],pcls[29],pdps[29],vec2(4.,2.)),

    //   Track(pcls[24],pdps[24],pcls[25],pdps[25],pcls[30],pdps[30],pcls[31],pdps[31],vec2(-4.,4.)),
    //   Track(pcls[25],pdps[25],pcls[26],pdps[26],pcls[31],pdps[31],pcls[32],pdps[32],vec2(-2.,4.)),
    //   Track(pcls[26],pdps[26],pcls[27],pdps[27],pcls[32],pdps[32],pcls[33],pdps[33],vec2(0.,4.)),
    //   Track(pcls[27],pdps[27],pcls[28],pdps[28],pcls[33],pdps[33],pcls[34],pdps[34],vec2(2.,4.)),
    //   Track(pcls[28],pdps[28],pcls[29],pdps[29],pcls[34],pdps[34],pcls[35],pdps[35],vec2(4.,4.))
    // );
    // float w = 0.;
    // vec2 p = vec2(0.,0.);
    // for(int i = 0; i < 25; i++){
    //   track(w,p,center,trs[i]);
    // }
    // vec2 pf = vUV + p / max(w,1.) * ts;
    // float pw = float(w > 0.);
    // vec3 pfc = texture(prevColorSampler, vUV).rgb * pw;

    vec2 centCoord = vUV * 2. -1.;
    vec3 rp = vec3(centCoord,depth);
    vec3 sRayPos = vec3(0.,0.,0.);
    vec3 sRayDir = normalize(rp-sRayPos);
    float sRayLen = length(rp);
    vec3 rayPos = sRayPos;
    vec3 rayDir = sRayDir;
    float rayLen = sRayLen;
    vec3 color = vec3(1.);
    vec3 fColor = vec3(0.);
    float at = 1.;
    float bt = 1.;
    int j = 0;
    for(float i = 0.;i < 16.;i++,j++){
      vec3 rayEnd = rayPos + rayDir * rayLen;
      float rd = pow(zScale,texture(depthSampler, rayEnd.xy*.5+.5).r*16.)/16.;
      vec3 crNorm = ((texture(normalSampler,rayEnd.xy*.5+.5).rgb)).rgb;
      bool invF = dot(crNorm,rayDir) < .0;
      if((invF && abs(rd-rayEnd.z) < rd*.1)||all(equal(sRayPos,rayPos))){
        vec3 difC = texture(textureSampler,rayEnd.xy*.5+.5).rgb*1.1;
        vec3 diff = pow(difC,vec3(4.))+difC;
        color = color*diff;
        at *= .8;
        rayPos = rayEnd;
        rayDir = normalize(reflect(-rayDir,crNorm)+noiseness*vec3(random(vUV+Time+i),random(vUV+Time+i+1.),random(vUV+Time+i-1.)));
        rayLen = .15;
        j = 0;
      }else{
        float tt = (rayEnd.z - rd);
        rayLen += tt;
      }
      if(at < .1 || rayLen < 0. || rayEnd.x < -1. || rayEnd.x > 1. || rayEnd.y < -1. || rayEnd.y > 1.||j > 2){
        fColor = mix(fColor,color,1./bt);
        color = vec3(1.);
        bt++;
        at = 1.;
        rayPos = sRayPos;
        rayDir = sRayDir;
        rayLen = sRayLen;
        j = 0;
      }
    }        
    fColor = mix(fColor,color,1./bt);
    vec3 finalR = fColor;//mix(max(pfc,fColor),fColor,.05);
    gl_FragColor = vec4(finalR,1.);
}

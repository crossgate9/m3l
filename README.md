# M3L, MCDE

## Usage
<pre>
git clone --recursive https://github.com/crossgate9/m3l

%% in matlab command window
run('matlab/main.m');
run('vlfeat/toolbox/vl_setup');
</pre>

## Packages

The source codes contains the following image feature extraction algorithm:

- Color Hist
- Local Binary Pattern(LBP)

All above codes only support simple usage. To further extract the image features, we recommend to use the external package to do the job: 

- [VLFeat](http://www.vlfeat.org/install-matlab.html "VLFeat")   
The package is included as submodule of this repository. For more information, please check VLFeat homepage.
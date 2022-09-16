# fast-image-level-vote

code for paper：A high-resolution feature network image-level classification method for hyperspectral image， Acta Geodaetica et Cartographica Sinica (AGCS) 

# Requirements
os
argparse
time
numpy
torch
datetime
sklearn

# Usagy

We provide a demo of the Salinas hyperspectral data by run the file of train_fastnet_vote_SA.py. The data is put in the realease, you need to download it and put it into the HSI_data file. If you want to run the code in your own data, you can accordingly change the input and tune the parameters. Please refer to the paper for more details.


# License
Copyright (C) 2022 Yifan Sun

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.



# References
[1]  @article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
  year={2019}
}

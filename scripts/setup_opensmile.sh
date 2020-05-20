#!/bin/bash
cd ./tools
tar -xvzf ./opensmile-2.3.0.tar.gz
cp -r ./gemaps_config/gemaps_50ms ./opensmile-2.3.0/config/
cp -r ./gemaps_config/shared ./opensmile-2.3.0/config/

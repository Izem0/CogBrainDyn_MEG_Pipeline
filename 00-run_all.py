#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:37:45 2019

@author: ie258305
"""

exec(open("/volatile/Repos/CogBrainDyn_MEG_Pipeline/01-import_and_filter.py").read())
exec(open("/volatile/Repos/CogBrainDyn_MEG_Pipeline/02-apply_maxwell_filter.py").read())
exec(open("/volatile/Repos/CogBrainDyn_MEG_Pipeline/03-extract_events.py").read())
exec(open("/volatile/Repos/CogBrainDyn_MEG_Pipeline/04-make_epochs.py").read())
exec(open("/volatile/Repos/CogBrainDyn_MEG_Pipeline/05a-run_ica.py").read())
exec(open("/volatile/Repos/CogBrainDyn_MEG_Pipeline/06a-apply_ica.py").read())
exec(open("/volatile/Repos/CogBrainDyn_MEG_Pipeline/07-make_evoked.py").read())
exec(open("/volatile/Repos/CogBrainDyn_MEG_Pipeline/07-m-visualise_evoked.py").read())
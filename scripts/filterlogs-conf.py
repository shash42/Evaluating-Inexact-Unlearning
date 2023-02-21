import re, sys, json

err_reg = r'-eval(-debug-last)?:(\w+)_Model_D_(\w+) ->.*, Error:(\S+)'
confmod_reg = r'Forgetting performance of (\w+)'
conf_reg = r'-eval(-debug-last)?:(\w+):\tConf_Score = (\w+)'
CRT_reg = r'-eval(-debug-last)?:(\w+):\t._FP = (\w+)\t._FN = (\w+)'
MIA_reg = r'-eval(-debug-last)?:(\S+)-Confidence - Shadow: (\S+)\tTarget: (\S+)'

def generate_output(model, res, path):
    d = {}
    d.update({'Model':model, 'Score-Re':-1000, 'Score-F':-1000, 'Score-Te':-1000,\
        'Err-Te_f':-1000, 'Err-F':-1000, 'Err-T':-1000, 'Err-Re':-1000, 'Err-Te_r':-1000,\
        'MIA-Conf-Sha':-1000, 'MIA-Conf-Tar':-1000, 'MIA-Sha':-1000, 'MIA-Tar':-1000})

    print(f'Model: {model}')
    d['Model'] = model
    if 'Retain' in res.keys(): 
        print(f"Score-Re: {float(res['Retain'])}")
        d['Score-Re'] = float(res['Retain'])
    if 'Forget' in res.keys():
        print(f"Score-F: {float(res['Forget'])}")
        d['Score-F'] = float(res['Forget'])
    if 'Test' in res.keys(): 
        print(f"Score-Te: {float(res['Test'])}")
        d['Score-Te'] = float(res['Test'])

    print(f"Err-F: {float(res['f'])*100}")
    d['Err-F'] = float(res['f'])*100
    if 'te_f' in res.keys():
        print(f"Err-Te_f: {float(res['te_f'])*100}")
        d['Err-Te_f'] = float(res['te_f'])*100
    if 't' in res.keys():
        print(f"Err-T: {float(res['t'])*100}")
        d['Err-T'] = float(res['t'])*100
    print(f"Err-Re: {float(res['r'])*100}")
    d['Err-Re'] = float(res['r'])*100
    if 'te_r' in res.keys(): 
        print(f"Err-Te_r: {float(res['te_r'])*100}")
        d['Err-Te_r'] = float(res['te_r'])*100

    if 'Exch-MIA' in res.keys(): 
        print(f"MIA-Conf-Shadow: {float(res['Exch-MIA'][0])*100}\tMIA-Conf-Target: {float(res['Exch-MIA'][1])*100}")
        d['MIA-Conf-Sha'] = float(res['Exch-MIA'][0])*100
        d['MIA-Conf-Tar'] = float(res['Exch-MIA'][1])*100
    if 'MIA' in res.keys(): 
        print(f"MIA-Shadow: {float(res['MIA'][0])*100}\tMIA-Target: {float(res['MIA'][1])*100}")
        d['MIA-Sha'] = float(res['MIA'][0])*100
        d['MIA-Tar'] = float(res['MIA'][1])*100
    print()
    
    with open(f'{path}-{model}.json', 'w', encoding='utf-8') as f:
        json.dump(d, f, sort_keys=True, indent=4, separators=(',', ': '))

def readLog(log_file_path):
    results = {}
    MIAmatch = None
    with open(log_file_path, "r") as file:
        for line in file:
            
            #Get MIAScores
            tmpmatch = re.search(MIA_reg, line)
            if tmpmatch is not None:
                # print(tmpmatch)
                MIAmatch = tmpmatch

            #Get Errors
            match = re.search(err_reg, line)
            if match is not None:
                if match[2] not in results.keys():
                    #print(match[2], match[3], match[4])
                    results.update({match[2] : {}})
                results[match[2]].update({match[3]:match[4]})
                if MIAmatch is not None:
                    results[match[2]].update({MIAmatch[2]:(MIAmatch[3], MIAmatch[4])})
            
            #Get ConfMod
            match = re.search(confmod_reg, line)
            if match is not None:
                curr_mod = match[1]
                if MIAmatch is not None:
                    results[curr_mod].update({MIAmatch[2]:(MIAmatch[3], MIAmatch[4])})
            
            #Get ConfScores
            match = re.search(conf_reg, line)
            if match is not None:
                results[curr_mod].update({match[2]:match[3]})

            #Get CRTScores
            match = re.search(CRT_reg, line)
            if match is not None:
                results[curr_mod].update({match[2]:int(match[3])+int(match[4])})
    
    return results

if __name__ == "__main__":

    while(True):
        log_file_path = input()   # Get the input
        if log_file_path == "":       # If it is a blank line...
            break           # ...break the loop
        log_file_path = log_file_path.replace("%20", " ")
        log_file_path = log_file_path.replace("file:///home/shashwat/Documents/MachineUnlearning/Codebase/", "")
        curr_mod = ""
        results = readLog(log_file_path)

        for model, res in results.items():
            generate_output(model, res, log_file_path)

        
# change "<中原地产>首席分析师[张大伟]说" to "中原地产首席分析师张大伟说"

def delete(input_file, out_file):
    special_tokens = ['<', '>', '[', ']', '(', ')']
    res = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            n_line, txt = line.strip().split()
            for sptoken in special_tokens:
                txt = txt.replace(sptoken, '')
            res.append(n_line+' '+txt)

    with open(out_file, 'w') as f:
        for n_line in res:
            f.write(n_line + '\n')

    

    

if __name__ == "__main__":
    input_file = 'end2end-spner.pred'
    out_file = 'end2end-sp.pred'
    delete(input_file, out_file)
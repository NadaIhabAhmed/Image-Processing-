# --------------------VERILOG-----------------------------
def verilog(equation,output):
    input = ''
    for element in equation:
        input = input + element
    equ = input
    input = input.replace(' ( ', '')
    input = input.replace(' ) ', '')
    input = input.replace(' = ', '')
    input = input.replace(' & ', '')
    input = input.replace(' | ', '')
    input = input[2:]
    inpu = ''
    for element in input:
        inpu = inpu + element + ', '
    code = "module verilog_gates( input " + inpu + "; output " + output + ");"
    code2 = "\nassign " + equ
    code3 = "\nendmodule"

    f = open("verilog.v", "w")
    f.write(code + code2 + code3)
    f.close()
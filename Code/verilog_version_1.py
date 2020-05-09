def verilog(equation, output):
    string_of_input = str(equation).split(" = , ( )  | ")
    code = "module verilog_gates( input " + "; output " + equation[0] + ");"
    code2 = "\nassign " + str(equation)
    code3 = "\nendmodule"

    f = open("verilog.v", "w")
    f.write(code + code2 + code3)
    f.close()
	
	
'''
equation: e =  (  (  ( k | i )  &  ( c & f )  )  &  ( b | a )  ) 
output = [e, k, i, c, f, b, a]
'''
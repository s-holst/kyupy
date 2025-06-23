module gates (a, b, c, o0, o1, o2, o3 );
input a;
input b;
input c;
output o0;
output o1;
output o2;
output o3;

AND2_X1 andgate (.A1 ( a ) , .A2 ( b ) , .ZN ( o0 ) ) ;
NAND2_X1 nandgate (.A1 ( a ) , .A2 ( b ) , .ZN ( o1 ) ) ;
OAI21_X1 oai21gate (.B1(a), .B2(b), .A(c), .ZN(o2) ) ;
MUX2_X1 mux2gate (.A(a), .B(b), .S(c), .Z(o3)) ;

endmodule
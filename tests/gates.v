module gates (a, b, o0, o1 );
input a;
input b;
output o0;
output o1;

AND2X1 andgate (.IN1 ( a ) , .IN2 ( b ) , .Q ( o0 ) ) ;
NAND2X1 nandgate (.IN1 ( a ) , .IN2 ( b ) , .QN ( o1 ) ) ;


endmodule
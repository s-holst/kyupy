/* Generated by Yosys 0.9 (git sha1 UNKNOWN, gcc 4.8.5 -fPIC -Os) */

(* top =  1  *)
(* src = "rng_haltonBase2.v:1" *)
module rng1(clk, reset, o_output);
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:19" *)
  wire [11:0] _00_;
  wire _01_;
  wire _02_;
  wire _03_;
  wire _04_;
  wire _05_;
  wire _06_;
  wire _07_;
  wire _08_;
  wire _09_;
  wire _10_;
  wire _11_;
  wire _12_;
  wire _13_;
  wire _14_;
  wire _15_;
  wire _16_;
  wire _17_;
  wire _18_;
  wire _19_;
  wire _20_;
  wire _21_;
  wire _22_;
  wire _23_;
  wire _24_;
  wire _25_;
  wire _26_;
  wire _27_;
  wire _28_;
  wire _29_;
  wire _30_;
  wire _31_;
  wire _32_;
  wire _33_;
  wire _34_;
  (* src = "rng_haltonBase2.v:2" *)
  input clk;
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:12" *)
  wire \halton.clk ;
  (* init = 12'h000 *)
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:17" *)
  wire [11:0] \halton.counter ;
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:14" *)
  wire [11:0] \halton.o_output ;
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:13" *)
  wire \halton.reset ;
  (* src = "rng_haltonBase2.v:4" *)
  output [11:0] o_output;
  (* src = "rng_haltonBase2.v:3" *)
  input reset;
  AND2X1 _35_ (
    .IN1(\halton.counter [1]),
    .IN2(\halton.counter [0]),
    .Q(_01_)
  );
  NOR2X0 _36_ (
    .IN1(\halton.counter [1]),
    .IN2(\halton.counter [0]),
    .QN(_02_)
  );
  NOR3X0 _37_ (
    .IN1(reset),
    .IN2(_01_),
    .IN3(_02_),
    .QN(_00_[1])
  );
  AND2X1 _38_ (
    .IN1(\halton.counter [2]),
    .IN2(_01_),
    .Q(_03_)
  );
  NOR2X0 _39_ (
    .IN1(\halton.counter [2]),
    .IN2(_01_),
    .QN(_04_)
  );
  NOR3X0 _40_ (
    .IN1(reset),
    .IN2(_03_),
    .IN3(_04_),
    .QN(_00_[2])
  );
  AND4X1 _41_ (
    .IN1(\halton.counter [1]),
    .IN2(\halton.counter [0]),
    .IN3(\halton.counter [2]),
    .IN4(\halton.counter [3]),
    .Q(_05_)
  );
  NOR2X0 _42_ (
    .IN1(\halton.counter [3]),
    .IN2(_03_),
    .QN(_06_)
  );
  NOR3X0 _43_ (
    .IN1(reset),
    .IN2(_05_),
    .IN3(_06_),
    .QN(_00_[3])
  );
  AND2X1 _44_ (
    .IN1(\halton.counter [4]),
    .IN2(_05_),
    .Q(_07_)
  );
  NOR2X0 _45_ (
    .IN1(\halton.counter [4]),
    .IN2(_05_),
    .QN(_08_)
  );
  NOR3X0 _46_ (
    .IN1(reset),
    .IN2(_07_),
    .IN3(_08_),
    .QN(_00_[4])
  );
  AND2X1 _47_ (
    .IN1(\halton.counter [5]),
    .IN2(_07_),
    .Q(_09_)
  );
  NOR2X0 _48_ (
    .IN1(\halton.counter [5]),
    .IN2(_07_),
    .QN(_10_)
  );
  NOR3X0 _49_ (
    .IN1(reset),
    .IN2(_09_),
    .IN3(_10_),
    .QN(_00_[5])
  );
  AND4X1 _50_ (
    .IN1(\halton.counter [4]),
    .IN2(\halton.counter [5]),
    .IN3(\halton.counter [6]),
    .IN4(_05_),
    .Q(_11_)
  );
  NOR2X0 _51_ (
    .IN1(\halton.counter [6]),
    .IN2(_09_),
    .QN(_12_)
  );
  NOR3X0 _52_ (
    .IN1(reset),
    .IN2(_11_),
    .IN3(_12_),
    .QN(_00_[6])
  );
  AND2X1 _53_ (
    .IN1(\halton.counter [7]),
    .IN2(_11_),
    .Q(_13_)
  );
  NOR2X0 _54_ (
    .IN1(\halton.counter [7]),
    .IN2(_11_),
    .QN(_14_)
  );
  NOR3X0 _55_ (
    .IN1(reset),
    .IN2(_13_),
    .IN3(_14_),
    .QN(_00_[7])
  );
  AND3X1 _56_ (
    .IN1(\halton.counter [7]),
    .IN2(\halton.counter [8]),
    .IN3(_11_),
    .Q(_15_)
  );
  NOR2X0 _57_ (
    .IN1(\halton.counter [8]),
    .IN2(_13_),
    .QN(_16_)
  );
  NOR3X0 _58_ (
    .IN1(reset),
    .IN2(_15_),
    .IN3(_16_),
    .QN(_00_[8])
  );
  AND4X1 _59_ (
    .IN1(\halton.counter [7]),
    .IN2(\halton.counter [8]),
    .IN3(\halton.counter [9]),
    .IN4(_11_),
    .Q(_17_)
  );
  NOR2X0 _60_ (
    .IN1(\halton.counter [9]),
    .IN2(_15_),
    .QN(_18_)
  );
  NOR3X0 _61_ (
    .IN1(reset),
    .IN2(_17_),
    .IN3(_18_),
    .QN(_00_[9])
  );
  AND2X1 _62_ (
    .IN1(\halton.counter [10]),
    .IN2(_17_),
    .Q(_19_)
  );
  NOR2X0 _63_ (
    .IN1(\halton.counter [10]),
    .IN2(_17_),
    .QN(_20_)
  );
  NOR3X0 _64_ (
    .IN1(reset),
    .IN2(_19_),
    .IN3(_20_),
    .QN(_00_[10])
  );
  AND3X1 _65_ (
    .IN1(\halton.counter [10]),
    .IN2(\halton.counter [11]),
    .IN3(_17_),
    .Q(_21_)
  );
  AOI21X1 _66_ (
    .IN1(\halton.counter [10]),
    .IN2(_17_),
    .IN3(\halton.counter [11]),
    .QN(_22_)
  );
  NOR3X0 _67_ (
    .IN1(reset),
    .IN2(_21_),
    .IN3(_22_),
    .QN(_00_[11])
  );
  NOR2X0 _68_ (
    .IN1(reset),
    .IN2(\halton.counter [0]),
    .QN(_00_[0])
  );
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:19" *)
  DFFX1 _69_ (
    .CLK(clk),
    .D(_00_[0]),
    .Q(\halton.counter [0]),
    .QN(_23_)
  );
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:19" *)
  DFFX1 _70_ (
    .CLK(clk),
    .D(_00_[1]),
    .Q(\halton.counter [1]),
    .QN(_24_)
  );
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:19" *)
  DFFX1 _71_ (
    .CLK(clk),
    .D(_00_[2]),
    .Q(\halton.counter [2]),
    .QN(_25_)
  );
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:19" *)
  DFFX1 _72_ (
    .CLK(clk),
    .D(_00_[3]),
    .Q(\halton.counter [3]),
    .QN(_26_)
  );
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:19" *)
  DFFX1 _73_ (
    .CLK(clk),
    .D(_00_[4]),
    .Q(\halton.counter [4]),
    .QN(_27_)
  );
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:19" *)
  DFFX1 _74_ (
    .CLK(clk),
    .D(_00_[5]),
    .Q(\halton.counter [5]),
    .QN(_28_)
  );
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:19" *)
  DFFX1 _75_ (
    .CLK(clk),
    .D(_00_[6]),
    .Q(\halton.counter [6]),
    .QN(_29_)
  );
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:19" *)
  DFFX1 _76_ (
    .CLK(clk),
    .D(_00_[7]),
    .Q(\halton.counter [7]),
    .QN(_30_)
  );
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:19" *)
  DFFX1 _77_ (
    .CLK(clk),
    .D(_00_[8]),
    .Q(\halton.counter [8]),
    .QN(_31_)
  );
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:19" *)
  DFFX1 _78_ (
    .CLK(clk),
    .D(_00_[9]),
    .Q(\halton.counter [9]),
    .QN(_32_)
  );
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:19" *)
  DFFX1 _79_ (
    .CLK(clk),
    .D(_00_[10]),
    .Q(\halton.counter [10]),
    .QN(_33_)
  );
  (* src = "rng_haltonBase2.v:7|rng_haltonBase2.v:19" *)
  DFFX1 _80_ (
    .CLK(clk),
    .D(_00_[11]),
    .Q(\halton.counter [11]),
    .QN(_34_)
  );
  assign \halton.clk  = clk;
  assign \halton.o_output  = { \halton.counter [0], \halton.counter [1], \halton.counter [2], \halton.counter [3], \halton.counter [4], \halton.counter [5], \halton.counter [6], \halton.counter [7], \halton.counter [8], \halton.counter [9], \halton.counter [10], \halton.counter [11] };
  assign \halton.reset  = reset;
  assign o_output = { \halton.counter [0], \halton.counter [1], \halton.counter [2], \halton.counter [3], \halton.counter [4], \halton.counter [5], \halton.counter [6], \halton.counter [7], \halton.counter [8], \halton.counter [9], \halton.counter [10], \halton.counter [11] };
endmodule
.phony: all gif

all:
	cargo build

gif:
	convert -monitor -delay 5 -loop 0 result_*.png result.gif

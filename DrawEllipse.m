function DrawEllipse(x, y, xR, yR)

t=-pi:0.01:pi;
x=x+xR*cos(t);
y=y+yR*sin(t);
plot(x,y, 'MarkerFaceColor', 'r');
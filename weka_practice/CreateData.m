a=-5;
b=5;
matrix=[];
 for i=1 : 6000
x1= a + (b-a)* rand (1);
x2= a + (b-a)* rand (1);
class=sign (-2+x1+2 * x2);
matrix=[matrix; x1 x2 class];

end
resolve = ['X1','X2','class';matrix]
 save -ascii dataset.dat resolve
 dlmwrite('dataset.csv', resolve, ",")

%% Task 1 
%
clear, clc;
a = -10; b = 20; n = 1; 
xGrid = [a : n : b];
yGrid = cos(xGrid.*xGrid - 4*abs(xGrid));
plot (xGrid,yGrid);

[yMin, xIndex] = min(yGrid);
xMin = xGrid(1,xIndex);
disp(['Minimum: x = ', num2str(xMin), ' y = ', num2str(yMin)]);
text(xMin, yMin, '\leftarrow min(y)');
[yMax, xIndex] = max(yGrid);
xMax = xGrid(1,xIndex);
disp(['Maximum: x = ', num2str(xMax), ' y = ', num2str(yMax)]);
text(xMax, yMax, '\leftarrow max(y)');

figure(3)
ylim([yMin - 0.1, yMax + 0.1]);
grid on;
xlabel('x');
ylabel('y = cos(x^2 - 4*|x|)');
title('Plot for the task 1');

%
%% Task 2 
%
clear, clc;
n = input ('Введите простое число\n');
if (isnan(n))
    disp('Input is not a number.');
elseif (n == inf)
    disp('Input is too big.');
elseif (isprime(n))
    disp('Input is a prime number.');
    DividedBy7 = 7:14:n; % 2.1
    TmpMatr = (2:1:n+1)';
    X = TmpMatr*ones(1, n); % 2.2
    c = 1:1:(n+1)*(n+1);
    B = reshape (c, n+1, n+1)';
    D = B(:, n:n+1);
else 
    disp('Input is not a prime number.');
end


%
%% Task 3
%
clear, clc;
Matr = (rand(5, 8) - 0.5)*20;
DiagMatr = diag(Matr)
maxDiag = max(DiagMatr)
S = sum(Matr')
P = prod(Matr')
relation = S./P
maxRelation = max(relation)
minRelation = min(relation)
SortedMatr = sortrows(Matr, 'descend')

%
%% Task 4
%
% B - пересечение 2, 4, ..., n - mod(n,2) столбцов и 
%                 1, 3, ..., m - (1 - mod(m,2)) строк
% R - пересечение 1, 3, ..., n - (1 - mod(n,2)) столбцов и 
%                 2, 4, ..., m - mod(m,2) строк
% G - пересечение 1, 3, ..., n - (1 - mod(n,2)) столбцов и
%                 2, 4, ..., m - mod(m,2) строк
% B, R  могут получиться из любой мтрицы. 
% Ограничения на G: равное кол-во четных и нечетных эл-тов в строке
clear, clc;
n = 5; m = 8; 
% n = input ('Введите количество строк');
% m = input ('Введите количество столбцов');
if (mod(m, 2) == 1)
    disp('Указанное разбиение построить нельзя')
else
    vector = 1:1:n*m;
    A = reshape (vector, [m, n])';
    B = A(2:2:n, 1:2:m);
    R = A(1:2:n, 2:2:m);
    G1 = A(1:1:n, 1:2:m);
    G2 = A(1:1:n, 2:2:m);
    G1(2:2:n, :) = 0;
    G2(1:2:n, :) = 0;
    G = G1+G2;
end


%
%% Task 5
%
clear, clc;
m = 10; n = 5;
X = 1:1:m ;
X1 = diag(X) ;
X2 = ones(m, n) ;
X3 = X1*X2;
FinalX = reshape(X3, 1, m*n);

Y = n:-1:1 ;
Y1 = diag(Y) ;
Y2 = ones(n, m) ;
Y3 = Y1*Y2;
FinalY = reshape(Y3', 1, m*n);

Final = [FinalX; FinalY]';
%
%% Task 6
%
clear, clc;
n = 5;
Input = [1:1:n]
InputCoord = [Input; Input*2; Input- 1]
X = InputCoord(1, :)
Y = InputCoord(2, :)
Z = InputCoord(3, :)

% first coordinate of a vector product
% Aij = yizj - yjzi 
A1 = Y.*Z'   % yjzi
A2 = A1'     % yizj
A = A2 - A1

% second coordinate of a vector product
% Bij = xizj - xizj
B1 = X.*Z'   % xjzi
B2 = B1'     % xizj
B = B2 - B1

% third coordinate of a vector product
% Cij = xiyj - xjyi
C1 = X.*Y'   % xjyi
C2 = C1'     % xiyj
C = C2 - C1

Ans = sqrt(A.*A + B.*B + C.*C) 
%
%% Task 7
%
clear, clc;
n = 5; m = 10;
a = round(rand(1, n)*10, 1) % [a1, a2, ..., an]
b = round(rand(1, m)*10, 1) % [b1, b2, ..., bn]
result = max ([(max(a) - min(b)), (max(b) - min(a))])
%
%% Task 8
%
clear, clc;
n = 10; k = 5;
A = reshape(1:1:n*k, k, n);

B = repmat(A, 1, n);    % spread horizontally, k*n^2
A = repmat(A, n, 1);    % spread vertically, k^2*n
A = reshape(A, k, n*n); % shape as B has
A = A - B;
Distances = sqrt(sum((A.*A)));
Distances = reshape(Distances, n, n);

%
%% Task 9
%
clear, clc;
step = 1; num_col = 10;
columns = 1:step:num_col;
time_myadd = zeros(1, num_col);
time_add = zeros(1, num_col);

num_time = 10;
res_time = zeros (1, num_time);
res_time1 = zeros (1, num_time);

for i = 1:num_col
    A = rand(columns(i),columns(i));
    B = rand(columns(i),columns(i));
    for j = 1:num_time
        tic();
        my_add(A,B);
        res_time(:, j) = toc();
        tic();
        A+B;
        res_time1(:, j) = toc();
    end
    time_myadd(i) = median(res_time);
    time_add(i) = median(res_time1);
end

figure(4)
plot (columns, time_myadd,'ko-', columns, time_add,'bo-');
title('Plot for the task 9');
xlabel('Number of columns in the square matrix');
ylabel('Time for computation');
legend('Time of myadd function','Time of default add function','Location','northwest');

%
%% Task 10
%
clear, clc;
n = 5; eps = 0.001;
A = round(rand(1, n)*10, 0)
RevA = A(end:-1:1)
if (abs(A - RevA) < eps) 
    disp('Вектор А симметричный')
else
    disp('Вектор А асимметричный')
end

%
%% Task 11
%
clear, clc;
n = 5; eps = 0.001; a = 10; b = 5;
A = rand(1, n)*a;
x = find(A > b); % ksi >= b
percentage = length(x)/length(A); % P(|ksi| >= x)
num_compare = a/(2*b); 
% E*|ksi| = int(x/a) from 0 to a 
%         = x^2/(2*a) from 0 to a = a/2
% E*|ksi|/x = (a/2)/b
if (percentage <= num_compare)
    disp ('The markov inequality holds')
else 
    disp ('The markov inequality does not hold')
end

%
%% Task 12
%
clear, clc;
iterations = 20;  % number of calculations of the primitive
dots = 100;       % number of dots used in calculations of the primitive
step = 0.01;      % for the grid in calculations of the primitive
a = -10;
b = a + dots*step;
X_primitive = a + dots*step/2 : dots*step : a + iterations*dots*step;

Y_primitive = zeros(1, iterations);
Y_prim_simp = zeros(1, iterations);
Y_prim_rect = zeros(1, iterations);

for i = 1:iterations
    X = a: step: b;
    Y = exp(-X.*X);
    Y_primitive(i) = trapz(X, Y);
    Y_prim_simp(i) = simpson (X, Y);
    Y_prim_rect(i) = rectangles (X, Y);
    
    a = b;
    b = a + dots*step;
end

% calculating convergence 
stepmin = 0.001; stepmax = 1;
h = stepmin:0.001:stepmax;
n = length(h);
Xmin = -10; Xmax = 10;
num_tries = 10; % how many times do we measure the time
res_time = zeros (1, num_tries);
res_time1 = zeros (1, num_tries);
Y_trap_time1 = zeros (1, n);
Y_trap_time2 = zeros (1, n);
Y_rect_time1 = zeros (1, n);
Y_rect_time2 = zeros (1, n);
Y_simp_time1 = zeros (1, n);
Y_simp_time2 = zeros (1, n);
Y_trap_conv = zeros (1, n);
Y_simp_conv = zeros (1, n);
Y_rect_conv = zeros (1, n);

for i = 1:n
    X = Xmin:(h(1, i)/2):Xmax; 
    Y = exp(-X.*X);
    X1 = X(1:2:end);
    Y1 = Y(1:2:end);
    
    for j = 1:num_tries
        tic();
        Y_trap_conv(:, i) = trapz(X, Y);
        res_time(:, j) = toc();
        tic();
        Y_trap_conv(:, i) = Y_trap_conv(:, i) - trapz(X1, Y1);
        res_time1(:, j) = toc();
    end
    Y_trap_time1(:,i) = median(res_time);
    Y_trap_time2(:,i) = median(res_time1);
        
    for j = 1:num_tries
        tic();
        Y_rect_conv(:, i) = rectangles(X, Y);
        res_time(:, j) = toc();
        tic();
        Y_rect_conv(:, i) = Y_rect_conv(:, i) - rectangles(X1, Y1);
        res_time1(:, j) = toc();
    end
    Y_rect_time1(:,i) = median(res_time);
    Y_rect_time2(:,i) = median(res_time1);
    
    for j = 1:num_tries
        tic();
        Y_simp_conv(:, i) = simpson(X, Y);
        res_time(:, j) = toc();
        tic();
        Y_simp_conv(:, i) = Y_simp_conv(:, i) - simpson(X1, Y1);
        res_time1(:, j) = toc();
    end
    Y_simp_time1(:,i) = median(res_time);
    Y_simp_time2(:,i) = median(res_time1);
end
    

% loglog time scale
n_log = 100;
h_log = logspace (-5, 1, n_log);
Y_trap_log_time = zeros (1, n_log);
Y_rect_log_time = zeros (1, n_log);
Y_simp_log_time = zeros (1, n_log);
for i = 1:n_log
    X = Xmin:(h_log(:,i)):Xmax; 
    Y = exp(-X.*X);
    
    for j = 1:num_tries
        tic();
        trapz(X, Y);
        res_time(:, j) = toc();
    end
    Y_trap_log_time(:,i) = median(res_time);
        
    for j = 1:num_tries
        tic();
        rectangles(X, Y);
        res_time(:, j) = toc();
    end
    Y_rect_log_time(:,i) = median(res_time);
    
    for j = 1:num_tries
        tic();
        simpson(X, Y);
        res_time(:, j) = toc();
    end
    Y_simp_log_time(:,i) = median(res_time);
end


figure(2);
plot (X_primitive, Y_primitive, X_primitive, Y_prim_rect, X_primitive, Y_prim_simp);
title ('Different types of integrations');
xlabel('Arguement');
ylabel('Primitive of a function f(x) = exp(-x^2)');
legend('trapz','rectangles','simpson', 'Location','northwest');

figure(1);
subplot(3,3,1);
plot(X_primitive, Y_primitive);
title('trapz');
subplot(3,3,2);
plot(X_primitive, Y_prim_rect);
title('rectangles');
subplot(3,3,3);   
plot(X_primitive, Y_prim_simp);
title('simpson');

% convergence
subplot(3,3,4);
plot(h, Y_trap_conv);
title('trapz convergence');
xlabel('step size');
subplot(3,3,5);
plot(h, Y_rect_conv);
title('rectangles conv.');
xlabel('step size');
subplot(3,3,6);   
plot(h, Y_simp_conv);
title('simpson conv.');
xlabel('step size');

% time
subplot(3,3,7);
plot(h(2:1:end), Y_trap_time1(2:1:end), h(2:1:end), Y_trap_time2(2:1:end));
title('trapz time');
xlabel('step size');
legend('h', 'h/2');
subplot(3,3,8);
plot(h, Y_rect_time1, h, Y_rect_time2);
title('rectangles time');
xlabel('step size');
legend('h', 'h/2');
subplot(3,3,9);   
plot(h, Y_simp_time1, h, Y_simp_time2);
title('simpson time');
xlabel('step size');
legend('h', 'h/2');

%log time
figure(6);
loglog(h_log, Y_trap_log_time, h_log, Y_rect_log_time, h_log, Y_simp_log_time);
title('Log-log scale plot of time for the task 12');
xlabel('step size');
legend('trapz', 'rectangles', 'simpson');

%
%% Task 13
%
clear, clc;
a = -20; b = 20; X = round(rand*10, 1); 
n = 300;
h = logspace(a,b, n);
Y = func13(X);
D = deriv13(X);

Central_der = (func13(X+h) - func13(X-h))./(2*h)
Right_der = (func13(X+h) - func13(X))./h
figure(5)
loglog(h, abs(D - Central_der), h, abs(D - Right_der))
%loglog(h, D, h, Central_der, h, Right_der);
title('Plot for the task 13')
xlabel('Arguement')
ylabel('Deviation of the difference derivative from the derivative')
legend('|Derivative - CentralDerivative|','|Derivative - RightDerivative|','Location','southeast');
%legend('Derivative', 'CentralDerivative','RightDerivative','Location','southeast');

% task 13: function and it's derivative 
function res = func13 (x)
    res = sin(x) - log(x)
end

function res = deriv13 (x)
    res = cos(x) - 1./x
end

% task 12: rectangles and simpson methods
% X = [x1, x2, x3, x4, x5] => Y_mid = [f(x2), f(x4)]
% X = [x1, x2, x3, x4] => Y_mid = [f(x2)]
function res = rectangles (X, Y)
    if (length(X) ~= length(Y))
        disp ('Incorrect size grid')
    else 
        n = length(X);
        step = (X(n) - X(1))/n;
        Y_mid = Y([2:2:n]);
        res = 2*step*sum(Y_mid)
    end
end

function res = simpson (X, Y)
    if (length(X) ~= length(Y))
        disp ('Incorrect size grid')
    else 
        n = length(X);
        step = (X(n) - X(1))/(n-1);
        Y_edges = Y([1:2:n]); % a = x1 < x3 < ... < x2n-1 = b
        Y_mid = Y([2:2:n-1]); % x2 < x4 < ... < x2n-2 = b
        res = step*(4*sum(Y_mid) + 2*sum(Y_edges) - Y(n) - Y(1))/3
        if (mod(n, 2) == 0)
            res = res + (Y(n) + Y(n-1))*step/2 
        end
    end
end

% Task 9
function C = my_add(A, B)
% This function returns the sum of 
% 2 matrices A and B 
    x = size(A)
    if (x == size(B))
        C = zeros ([x(1), x(2)]);
        for i = 1:x(1)
            for j = 1:x(2)
                C(i, j) = A(i, j) + B(i,j)
            end
        end
    else 
        disp('Unmatched dimensions');
    end 
end
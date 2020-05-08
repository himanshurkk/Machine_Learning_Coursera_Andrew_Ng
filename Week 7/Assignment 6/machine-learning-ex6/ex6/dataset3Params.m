function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
c_val = [0.01 0.03 0.1 1 3 10 30]' ;
s_val = [0.01 0.03 0.1 1 3 10 30]' ;

pre_error =  zeros(length(c_val), length(s_val));
result= zeros(length(c_val)+ length(s_val),3);
row=1;

for i=1:length(c_val)
  for j=1: length(s_val)
      c_test = c_val(i);
      s_test= s_val(j);
      
      
      
      model= svmTrain(X,y,c_test,@(x1,x2) gaussianKernel(x1,x2,s_test));
      
      predictions = svmPredict(model,Xval);
      pre_error(i,j) = mean(double(predictions~=yval));
     
      result(row,:)= [pre_error(i,j), c_test, s_test];
      
      row = row+1;
  endfor
endfor


sorted_result= sortrows(result,1);

C = sorted_result(1,2);
sigma = sorted_result(1,3); 




% =========================================================================

end

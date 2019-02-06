## loading the packages to be used
using Ipopt
using JuMP
using LinearAlgebra


## Loading the data files
pathNameData="C:\\Users\\shuvo\\Google Drive\\Research_Works_2018\\Nonconvex_ADMM_DRS_adaptive_stepsize\\Numerical_simulations\\3_SAT_Problem\\Data_sets_3_SAT_problem\\"
DataFileNames=readdir("C:\\Users\\shuvo\\Google Drive\\Research_Works_2018\\Nonconvex_ADMM_DRS_adaptive_stepsize\\Numerical_simulations\\3_SAT_Problem\\Data_sets_3_SAT_problem")
inputDataFilePaths=map(x->string(pathNameData,x),DataFileNames) # this contains all the subfolders containing the 3-SAT instances

##
#i=1;
#include(inputDataFilePaths[i])
# this will load three matrices into the memory
# A: the 3nxn matrix associated with the 3-SAT problem
# b: the 3n column vector associated with the 3-SAT problem
# sol: this is a solution associated with the three 3-SAT problem
#A
nIter=500;
## computing projection
function proj_3_SAT(x)
n=length(x);
y=zeros(n);
for i=1:n
    if x[i] > 0.5
        y[i]=1
    end
end
return y
end


## computing the proximal map
function prox_3_SAT(A,b,x)

    m,n = size(A)

    y_star=zeros(n);

    prox3SatModel = Model(solver = IpoptSolver(print_level=0));

    @variable(prox3SatModel, 0<= y[i=1:n] <= 1);

    @NLobjective(prox3SatModel, Min, sum((0.5*y[i]*y[i])+((-x[i])* y[i]) for i in 1:n));

    for i in 1:m
        @constraint(prox3SatModel, sum(A[i,j]*y[j] for j in 1:n) <= b[i]);
    end

    # print(proxBinModel, "\n");

    statusProx3SatModel = solve(prox3SatModel);


    if statusProx3SatModel == :Optimal
        y_star=getvalue(y);
    else
        error("Something is wrong :()");
    end

    return y_star;

end

## NC-ADMM iteration function
function NC_ADMM_3_SAT(x,y,z,A,b)
    x=prox_3_SAT(A,b,y-z)
    y=proj_3_SAT(x+z)
    z=z-y+x
    return x, y, z
end

## NC-DRS iteration function
function NC_DRS_3_SAT(x,y,z,A,b)
    x=prox_3_SAT(A,b,z)
    y=proj_3_SAT(2*x-z)
    z=z+y-x
    return x, y, z
end

## constraint_violation_count for a point in {0,1}^n
function n_violated_constraints(A,b,p)
    m,n=size(A)
    n_violated=0
    for i=1:m
        if dot(A[i,:],p) > b[i]
            n_violated=n_violated+1
        end
    end
     return n_violated
end


## NC-ADMM function
# returns
# number of violated constraints, best_point_found_so_far
function NC_ADMM(A,b,nIter)
    m,n=size(A)
    x=zeros(n)
    y=zeros(n)
    z=zeros(n)
    best_point=zeros(n)
    n_viol=n_violated_constraints(A,b,y)
    i=1
    while i <= nIter
        x,y,z=NC_ADMM_3_SAT(x,y,z,A,b)
        n_viol=n_violated_constraints(A,b,y)
        if n_viol == 0
           # println("best_point_found")
           best_point=y
           break
         else
           i=i+1
           continue
         end
     end
     if n_viol == 0
         return n_viol, best_point
     else
         return n_viol, y
     end
end



## NC-DRS function
# returns
# number of violated constraints, best_point_found_so_far
function NC_DRS(A,b,nIter)
    m,n=size(A)
    x=zeros(n)
    y=zeros(n)
    z=zeros(n)
    best_point=zeros(n)
    feasible_point_reached=0
    n_viol=n_violated_constraints(A,b,y)
    i=1
    while i <= nIter
        x,y,z=NC_DRS_3_SAT(x,y,z,A,b)
        n_viol=n_violated_constraints(A,b,y)
        if n_viol == 0
           # println("best_point_found")
           best_point=y
           break
         else
           i=i+1
           continue
         end
     end
     if n_viol == 0
         return n_viol, best_point
     else
         return n_viol, y
     end
end


## code for storing the output
for i2=1:10

    println("i2=",i2)

    inputDataFiles=readdir(inputDataFilePaths[i2])

    filesInCurrentFolder=map(x->string(inputDataFilePaths[i2],"\\",x),inputDataFiles)

    output_file_name=string(inputDataFilePaths[i2],"\\output_file.txt")

    outputFile = open(output_file_name,"w")

    for j2=1:10 #
        println("j2=",j2)
        include(filesInCurrentFolder[j2])
        write(outputFile,"## filename=")
        show(outputFile,filesInCurrentFolder[j2])
        write(outputFile,"\n \n")
        n_viol_DRS, best_sol_DRS = NC_DRS(A,b,nIter)
        write(outputFile,"n_viol_DRS=")
        show(outputFile, n_viol_DRS)
        write(outputFile,"\n \n")
        write(outputFile,"best_sol_DRS=")
        show(outputFile, best_sol_DRS)
        write(outputFile,"\n \n")
        n_viol_ADMM, best_sol_ADMM = NC_ADMM(A,b,nIter)
        write(outputFile,"n_viol_ADMM=")
        show(outputFile, n_viol_ADMM)
        write(outputFile,"\n \n")
        write(outputFile,"best_sol_ADMM=")
        show(outputFile, best_sol_ADMM)
        write(outputFile,"\n \n \n \n")
    end

    close(outputFile)
end

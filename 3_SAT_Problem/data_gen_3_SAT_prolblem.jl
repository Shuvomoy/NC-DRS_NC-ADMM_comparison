##This code generates data for the 3_SAT problem and writes the data into a Julia file
using Compat, Random, Distributions, LinearAlgebra, Plots, Statistics
using JuMP
using GLPKMathProgInterface
uniDst=Uniform(0,1);


##
function solve_3_SAT(A,b,m,n)
    sol=zeros(n)
    bipModel = Model(solver = GLPKSolverMIP())
    @variable(bipModel, x[1:n], Bin)
    @objective(bipModel, Min, 0)
    for i in 1:m
        @constraint(bipModel, sum(A[i,j]*x[j] for j in 1:n) <= b[i])
    end
    if solve(bipModel) == :Optimal
        println("solution through JuMP found")
        sol=getvalue(x)
        status=1
    else
        status=0
        println("problem infeasible!")
    end
    return sol, status
end

##

function data_gen_3_SAT_problemm(m,n)
    # m: number of clauses
    # n: number of variables
    A=zeros(m,n)
    b=zeros(m)
    while true
        for i in 1:m
            clause_vars=sample(1:n,3, replace=false);
            d=zeros(3)
            for j=1:3
                rho=rand(uniDst)
                if rho < 0.5
                    d[j]=1
                    A[i,clause_vars[j]]=1
                else
                    A[i,clause_vars[j]]=-1
                end
            end
            b[i]=sum(d)-1
        end
        sol, status = solve_3_SAT(A,b,m,n)
        if status == 1
            println("feasible instance found!")
            return A, b, sol
            break
        else
            println("infeasible instance, trying again")
        end
    end
end
##


#A,b,sol=data_gen_3_SAT_problemm(m,n)



##
#sol, status = solve_3_SAT(A,b,m,n)



n_array=10:10:100;
m_array=Int64.(3*n_array)


cd("C:\\Users\\shuvo\\Google Drive\\Research_Works_2018\\Nonconvex_ADMM_DRS_adaptive_stepsize\\Numerical_simulations\\3_SAT_Problem\\Data_sets_3_SAT_problem")


for i=1:length(m_array)
    m=m_array[i]
    n=n_array[i]
    for j=1:10
        A,b,sol=data_gen_3_SAT_problemm(m,n)
        file_name=string("data_3SAT_m",m,"_n",n,"inst_",j,".jl")
        outputFile=open(file_name,"w")
        write(outputFile,"A=")
        show(outputFile,A)
        write(outputFile,"; \n \n")
        write(outputFile,"b=")
        show(outputFile,b)
        write(outputFile,"; \n \n")
        write(outputFile,"sol=")
        show(outputFile,sol)
        write(outputFile,"; \n \n")
        close(outputFile)
    end
end

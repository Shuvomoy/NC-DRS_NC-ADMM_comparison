## All the modules
using Compat, Random, Distributions, LinearAlgebra, Plots, Statistics
Random.seed!(123) # setting the seed
M=1
nIter=100
## All the functions
function data_gen_sparse(m,M)
    n=2*m;
    nrmlDst=Normal(0,1);
    uniDst=Uniform(-M,M);
    A=rand(nrmlDst, m, n);
    k=convert(Int64,round(m/5));
    xhat=rand(uniDst,n);
    zerosIndex=sample(1:n,convert(Int64, round(n-k)), replace=false);
    xhat[zerosIndex].=0;
    sigma=LinearAlgebra.norm(A*xhat)/(20*sqrt(m));
    nrmlDstNoise=Normal(0,sigma);
    v=rand(nrmlDstNoise,m);
    b=A*xhat+v;
    return A, b, k, v, n, xhat
end


function proj_sparse(M,k,n,x)
    y=zeros(n);
    perm=sortperm(abs.(x),rev=true)
    for i in 1:n
        if i in perm[1:k]
            if x[i]>M
                y[i]=M
            elseif x[i]<-M
                y[i]=-M
            else
                y[i]=x[i]
            end
        end
    end
    return y
end


function prox_sparse(A,b,n,x)
    In=Matrix{Float64}(I, n, n);
    return inv(In+2*transpose(A)*A)*(x+2*transpose(A)*b)
end

function NC_ADMM_sparse_iteration(x,y,z,A,b,M,k,n)
    x=prox_sparse(A,b,n,y-z)
    y=proj_sparse(M,k,n,x+z)
    z=z-y+x
    return x, y, z
end

function NC_DRS_sparse_iteration(x,y,z,A,b,M,k,n)
    x=prox_sparse(A,b,n,z)
    y=proj_sparse(M,k,n,2*x-z)
    z=z+y-x
    return x, y, z
end

function NC_DRS(A,b,M,k,n, nIter)
    x=zeros(n)
    y=zeros(n)
    z=zeros(n)
    obj_DRS=zeros(nIter)
    best_point=zeros(n)
    obj_DRS[1]=(LinearAlgebra.norm(A*x-b))^2 # this is the error of the solution
    best_obj=obj_DRS[1]
    i=2
    while i<=nIter
        x,y,z=NC_DRS_sparse_iteration(x,y,z,A,b,M,k,n)
        obj_DRS_x=(LinearAlgebra.norm(A*x-b))^2
        obj_DRS_y=(LinearAlgebra.norm(A*y-b))^2
        if obj_DRS_x < best_obj
            best_obj=obj_DRS_x
            best_point=x
        end
        if obj_DRS_y < best_obj
            best_obj=obj_DRS_y
            best_point=y
        end
        obj_DRS[i]=min(obj_DRS_x,obj_DRS_y)
        i=i+1
    end
    return best_obj, best_point, obj_DRS
end

function NC_ADMM(A,b,M,k,n,nIter)
    x=zeros(n)
    y=zeros(n)
    z=zeros(n)
    best_point=zeros(n)
    obj_ADMM=zeros(nIter,1)
    obj_ADMM[1]=(LinearAlgebra.norm(A*x-b))^2 # this is the error of the solution
    best_obj=obj_ADMM[1]
    i=2
    while i<=nIter
        x,y,z=NC_ADMM_sparse_iteration(x,y,z,A,b,M,k,n)
        obj_ADMM_x=(LinearAlgebra.norm(A*x-b))^2
        obj_ADMM_y=(LinearAlgebra.norm(A*y-b))^2
        if obj_ADMM_x < best_obj
            best_obj=obj_ADMM_x
            best_point=x
        end
        if obj_ADMM_y < best_obj
            best_obj=obj_ADMM_y
            best_point=y
        end
        obj_ADMM[i]=min(obj_ADMM_x,obj_ADMM_y)
        i=i+1
    end
    return best_obj, best_point, obj_ADMM
end
##

function write_data(A,b,k,v,n, xhat,m, inst, outputFile)
    write(outputFile,string("# data for row size=",m," instance =",inst,"\n"))
    # writing A
    write(outputFile,"A=")
    show(outputFile,A)
    write(outputFile,"\n")
    # writing b
    write(outputFile,"b=")
    show(outputFile,b)
    write(outputFile,"\n")
    # wrting k
    write(outputFile,"k=")
    show(outputFile,k)
    write(outputFile,"\n")
    # writing v
    write(outputFile,"v=")
    show(outputFile,v)
    write(outputFile,"\n")
    # writing n
    write(outputFile,"n=")
    show(outputFile,n)
    write(outputFile,"\n")
    # writing xhat
    write(outputFile,"xhat=")
    show(outputFile,xhat)
    write(outputFile,"\n \n")
    write(outputFile,"## *************************")
    # all data written, now close the output file
end

#Time for running the master loop
m_array=50:150
mean_f_DRS=zeros(length(m_array))
mean_f_ADMM=zeros(length(m_array))
mean_dist_DRS=zeros(length(m_array))
mean_dist_ADMM=zeros(length(m_array))
n_sim_each_m=40

 cd("C:\\Users\\shuvo\\Google Drive\\Research_Works_2018\\Nonconvex_ADMM_DRS_adaptive_stepsize\\Numerical_simulations\\Sparse_Regression_Problem\\All_data")
outputFile = open("data_file_for_sparse_reg.jl","w")
##
for m in m_array
    # println("matrix size=", m, "x" , 2*m)
    best_f_DRS_m=zeros(n_sim_each_m)
    best_f_ADMM_m=zeros(n_sim_each_m)
    dist_best_point_xhat_DRS_m=zeros(n_sim_each_m)
    dist_best_point_xhat_ADMM_m=zeros(n_sim_each_m)
    for inst=1:n_sim_each_m
        println("# simulation for m= ", m, " instance number=", inst)
        A, b, k, v, n, xhat=data_gen_sparse(m,M)
        # write_data(A,b,k,v,n, xhat,m, inst, outputFile)
        best_obj_DRS, best_point_DRS, obj_DRS_vect = NC_DRS(A,b,M,k,n, nIter)
        println("# best_obj_DRS=",best_obj_DRS)
        best_f_DRS_m[inst]=best_obj_DRS
        dist_best_point_xhat_DRS_m[inst]=norm(best_point_DRS-xhat)
        println("distance_from_xstar_DRS",dist_best_point_xhat_DRS_m[inst])
        best_obj_ADMM, best_point_ADMM, obj_ADMM_vect = NC_ADMM(A,b,M,k,n, nIter)
        println("# best_obj_ADMM=",best_obj_ADMM)
        best_f_ADMM_m[inst]=best_obj_ADMM
        dist_best_point_xhat_ADMM_m[inst]=norm(best_point_ADMM-xhat)
        println("distance_from_xstar_ADMM",dist_best_point_xhat_ADMM_m[inst])
        println("###############################")
    end
    mean_f_DRS[m+1-m_array[1]]=mean(best_f_DRS_m)
    mean_f_ADMM[m+1-m_array[1]]=mean(best_f_ADMM_m)
    mean_dist_DRS[m+1-m_array[1]]=mean(dist_best_point_xhat_DRS_m)
    mean_dist_ADMM[m+1-m_array[1]]=mean(dist_best_point_xhat_ADMM_m)
end
close(outputFile)

##

## saving the output
 cd("C:\\Users\\shuvo\\Google Drive\\Research_Works_2018\\Nonconvex_ADMM_DRS_adaptive_stepsize\\Numerical_simulations\\Sparse_Regression_Problem\\All_data")
outputFile = open("output.jl","w")
write(outputFile,"mean_f_DRS=")
show(outputFile,mean_f_DRS)
write(outputFile,"\n \n")
write(outputFile,"mean_f_ADMM=")
show(outputFile,mean_f_ADMM)
write(outputFile,"\n \n")
write(outputFile,"mean_dist_DRS=")
show(outputFile,mean_dist_DRS)
write(outputFile,"\n \n")
write(outputFile,"mean_dist_ADMM=")
show(outputFile,mean_dist_ADMM)
write(outputFile,"\n \n")
close(outputFile)

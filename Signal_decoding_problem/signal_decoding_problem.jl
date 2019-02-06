##
using Compat, Random, Distributions, LinearAlgebra, Plots, Statistics
Random.seed!(123) # setting the seed
n=400
m=2000
nIter=10
total_inst=1000


##
# not to confuse with my iteration schemes I will set
# A:=H
# b:=y

function data_gen_signal_decoding(m,n)
    nrmlDst=Normal(0,1);
    H=rand(nrmlDst, m, n);
    x_star=rand([-3,-1,1,3],n)# rand([-3,-1,1,3],n)
    sigma=LinearAlgebra.norm(H*x_star)/(8*sqrt(m));
    nrmlDstNoise=Normal(0,sigma);
    v=rand(nrmlDstNoise,m);
    y=H*x_star+v;
    return H, y, v, x_star
end

function find_proj_i(z)
    fsbl_st=[-3,-1,1,3]#[-3,-1,1,3]
    proj_ind=indmin(abs.(z-fsbl_st))
    return fsbl_st[proj_ind]
end


function proj_signal_decoding(n,x)
    p=zeros(n)
    for i=1:n
        p[i]=find_proj_i(x[i])
    end
    return p
end

function prox_signal_decoding(A,b,n,x)
    In=Matrix{Float64}(I, n, n);
    p1=inv(In+2*transpose(A)*A)*(x+2*transpose(A)*b)
end

function NC_ADMM_signal_decoding(x,y,z,A,b,n)
    x=prox_signal_decoding(A,b,n,y-z)
    y=proj_signal_decoding(n,x+z)
    z=z-y+x
    return x, y, z
end

function NC_DRS_signal_decoding(x,y,z,A,b,n)
    x=prox_signal_decoding(A,b,n,z)
    y=proj_signal_decoding(n,2*x-z)
    z=z+y-x
    return x, y, z
end

function bit_error_rate(x_heur,x_star)
    n_corr=0
    n=length(x_heur)
    for i in 1:n
        if LinearAlgebra.norm(x_heur-x_star) <= 0.1
            n_corr=n_corr+1
        end
    end
    return (n-n_corr)/n
end


function NC_DRS(A,b,n,nIter,x_star)
    x=zeros(n)
    y=zeros(n)
    z=zeros(n)
    obj_DRS=zeros(nIter)
    best_point=zeros(n)
    obj_DRS[1]=(LinearAlgebra.norm(A*y-b))^2 # this is the error of the solution
    best_obj=obj_DRS[1]
    i=2
    while i<=nIter
        x,y,z=NC_DRS_signal_decoding(x,y,z,A,b,n)
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

function NC_ADMM(A,b,n,nIter,x_star)
    x=zeros(n)
    y=zeros(n)
    z=zeros(n)
    best_point=zeros(n)
    obj_ADMM=zeros(nIter,1)
    obj_ADMM[1]=(LinearAlgebra.norm(A*y-b))^2 # this is the error of the solution
    best_obj=obj_ADMM[1]
    i=2
    while i<=nIter
        x,y,z=NC_ADMM_signal_decoding(x,y,z,A,b,n)
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
function write_data(A,b, v, x_star, inst, outputFile)
    write(outputFile,string("## data for instance =",inst,"\n"))
    # writing A
    write(outputFile,"A=")
    show(outputFile,A)
    write(outputFile,"\n")
    # writing b
    write(outputFile,"b=")
    show(outputFile,b)
    write(outputFile,"\n")
    # writing v
    write(outputFile,"v=")
    show(outputFile,v)
    write(outputFile,"\n")
    # writing xhat
    write(outputFile,"x_star=")
    show(outputFile,x_star)
    write(outputFile,"\n \n")
    write(outputFile,"# *************************")
    # all data written, now close the output file
end

## time to run the simulation
best_f_DRS_m=zeros(total_inst)
best_f_ADMM_m=zeros(total_inst)
dist_best_point_xhat_DRS_m=zeros(total_inst)
dist_best_point_xhat_ADMM_m=zeros(total_inst)
#bit_error_rate_ADMM=zeros(total_inst)
#bit_error_rate_DRS=zeros(total_inst)
cd("C:\\Users\\shuvo\\Google Drive\\Research_Works_2018\\Nonconvex_ADMM_DRS_adaptive_stepsize\\Numerical_simulations\\Signal_decoding_problem\\All_data_signal_decoding")
outputFile = open("data_file_for_signal_decoding.jl","w")
for inst=1:total_inst
    println(" instance number=", inst)
    A, b, v,x_star=data_gen_signal_decoding(m,n)
    write_data(A,b, v, x_star, inst, outputFile)
    best_obj_DRS, best_point_DRS, obj_DRS_vect = NC_DRS(A,b, n, nIter,x_star)
    best_f_DRS_m[inst]=best_obj_DRS
    dist_best_point_xhat_DRS_m[inst]=norm(best_point_DRS-x_star)
    best_obj_ADMM, best_point_ADMM, obj_ADMM_vect = NC_ADMM(A,b, n, nIter, x_star)
    best_f_ADMM_m[inst]=best_obj_ADMM
    dist_best_point_xhat_ADMM_m[inst]=norm(best_point_ADMM-x_star)
end


close(outputFile)

# write the output
outputFile2 = open("output_file_for_signal_decoding.jl","w")
write(outputFile2,"best_f_DRS_m=")
show(outputFile2,best_f_DRS_m)
write(outputFile2,"; \n")
write(outputFile2,"best_f_ADMM_m=")
show(outputFile2,best_f_ADMM_m)
write(outputFile2,"; \n")
write(outputFile2,"dist_best_point_xhat_DRS_m=")
show(outputFile2,dist_best_point_xhat_DRS_m)
write(outputFile2,"; \n")
write(outputFile2,"dist_best_point_xhat_ADMM_m=")
show(outputFile2,dist_best_point_xhat_ADMM_m)
write(outputFile2,"; \n")
close(outputFile2)

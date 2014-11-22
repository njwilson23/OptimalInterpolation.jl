module OptimalInterpolation
import Distances

function distance_matrix(x1, x2)
    D = Array(Float64, (size(x1)[2], size(x2)[2]))
    for i = 1:size(D)[1]
        for j = 1:size(D)[2]
            D[i,j] = Distances.euclidean(x1[:,i], x2[:,j])
        end
    end
    return D
end

# Statistical functions to directly compute autocovariance and autocorrelation
demean(A::AbstractArray) = A - mean(A)
autocov{T}(X::Vector{T}) = T[mean(X[1:end-j].*X[j+1:end]) for j=0:length(X)-1]
autocorr(X::Vector) = autocov(X) / var(X)

# *model* is a fuction that reproduces the structure function at a lag *r*
function errorvariance(model::Function, C::AbstractArray, Ai::AbstractArray, Z::AbstractArray)
    ϵ = (var(Z) - 0.5*model(0)) * ones(length(Z))
    for i=1:length(ϵ)
        ϵ[i] -= sum(C[:,i] .* C[:,i]' .* Ai)
    end
    return ϵ
end

function oainterp(model::Function, Xi, X, Y; ϵ=1e-1)
    Yd = demean(Y)
    A = var(Yd) - 0.5*model(distance_matrix(X, X)) + diagm(ϵ*ones(length(X)))
    C = var(Yd) - 0.5*model(distance_matrix(X, Xi))

    Ainv = inv(A)
    α = Ainv*C
    Yi = α'*Yd + mean(Y)
    errv = errorvariance(model, C, Ainv, Yi)
    return Yi, errv
end

# Model estimation
function covariance_matrix(z1, z2)
    V = Array(Float64, (length(z1), length(z2)))
    for i = 1:size(V)[1]
        for j = 1:size(V)[2]
            V[i,j] = (z1[i] - z2[j])^2
        end
    end
    return V
end

function band_average(x::AbstractArray, y::AbstractArray, nbands::Int64)
    Ymean = Float64[]
    bands = linspace(minimum(x), maximum(x), nbands+1)
    for i=1:nbands
        n = 0
        ymean = 0.0
        for (x_, y_) in zip(x,y)
            if bands[i] < x_ < bands[i+1]
                ymean += y_
                n += 1
            end
        end
        if n != 0
            push!(Ymean, ymean/n)
        else
            push!(Ymean, NaN)
        end
    end
    return (Ymean, 0.5*[bands[1:end-1]+bands[2:end]])
end

# Computes a structure function *s*, defined is terms of correlation *r* as:
# s = 2σ^2 * (1 - r)
function structurefunc(x)
    r = autocorr(x)
    s = 2*var(x) * (1-r)
end

end # module

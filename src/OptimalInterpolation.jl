module OptimalInterpolation
import Distances

# Statistical functions to directly compute autocovariance and autocorrelation
demean(A::AbstractArray) = A - mean(A)
autocov{T}(X::Vector{T}) = T[mean(X[1:end-j].*X[j+1:end]) for j=0:length(X)-1]
autocorr(X::Vector) = autocov(X) / var(X)

# Some convenience functions
var_from_struct(s, Z::Vector) = var(Z) - 0.5*s
ones_like{T}(Z::AbstractArray{T}) = ones(T, size(Z))
function detrend(x, y)
    b = (x' * x) \ (x' * y)
    return b[1]*x
end

# Compute the pairwise euclidean distances between every element in two vectors
function distance_matrix(x1::AbstractArray, x2::AbstractArray)
    D = Array(Float64, (size(x1)[2], size(x2)[2]))
    for i = 1:size(D)[1]
        for j = 1:size(D)[2]
            D[i,j] = Distances.euclidean(x1[:,i], x2[:,j])
        end
    end
    return D
end
distance_matrix(x1::AbstractVector, x2::AbstractVector) = distance_matrix(x1', x2')

# Make an objective analysis (OA) estimate of the field known from observations
# at *X*, *Y* at the positions *Xi*. The field variance function is given by
# *model*, and the variance intrinsic to the observation is given by *ϵ0*
function oainterp(model::Function, Xi, X, Y; ϵ0=1e-1)
    Ym = mean(Y)
    Yd = Y - Ym
    A = model(distance_matrix(X, X)) + diagm(ϵ0*ones_like(Y))
    C = model(distance_matrix(X, Xi))

    Ainv = inv(A)
    α = Ainv*C
    Yi = α'*Yd + Ym
    ϵi = errorvariance(model, C, Ainv)
    return Yi, ϵi
end

# Compute the error variance from an OA estimate, given the modelled variance
# function *model*, prediction covariance matrix *C*, the inverted observation
# covariance matrix *Ainv*, and the observation data, *Z*.
function errorvariance(model::Function, C::AbstractArray, Ainv::AbstractArray)
    ϵ = model(0) * ones(size(C, 2))
    for i=1:length(ϵ)
        ϵ[i] -= sum(C[:,i] .* C[:,i]' .* Ainv)
    end
    return ϵ
end

#### Model estimation #####
# The following functions may be useful for estimating the structure function

# Compute the covariance matrix
function covariance_matrix(z1, z2)
    return demean(z1[:]) * demean(z2[:])'
end

# Band average *y* over *nbands* intervals in *x*
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
    return (0.5*[bands[1:end-1]+bands[2:end]], Ymean)
end

# Computes a structure function *s*, defined is terms of correlation *r* as:
# s = 2σ^2 * (1 - r)
function structurefunc(x)
    r = autocorr(x)
    s = 2*var(x) * (1-r)
end

end # module

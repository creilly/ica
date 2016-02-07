from scipy.misc import derivative
import numpy as np

delta = 1.e-4 # criterion for convergence

def negative_gaussian(x):
    return -1. * np.exp(-1. / 2. * np.square(x))

def log_gaussian(x):
    #can optimize a for better algorithm
    a = 1.5

    return 1./ a * np.log(np.cosh(a * x))

# converges each signal individually
def iterative():
    pass
# converges signals together
def symmetric():
    pass
# uses approximate for eigenvalue decomposition
def symmetric_approx():
    pass

# takes rectangular numpy array of with each row representing a signal.
# extracts original sources using independent component analysis
def extract_sources(signals,non_linear_function=negative_gaussian, method = 'iterative'):
    dimensions, samples = signals.shape

    # make each signal have zero mean
    centered = signals - np.average(signals,1).reshape(dimensions,1)    

    # compute expectation value of covariance matrix
    cov = np.cov(centered)

    # compute eigenvalue / eigenvector decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    E = eigenvectors

    # construct diagonal matrix of inverse square root of eigenvalues
    D = np.diag(1. / np.sqrt(eigenvalues))

    # apply linear transformation on centered signal data to have unit covariance
    whitened = E.dot(D).dot(E.transpose()).dot(centered)


    
    # first derivative of nonlinear nongaussianity-maximizing function
    def F(x):
        return derivative(non_linear_function, x, 1.e-8, 1)

    # second derivative
    def f(x):
        return derivative(non_linear_function, x, 1.e-8, 2)

    # Iterative counts 1 parameter at a time
    if method == 'iterative':
        print 'method is iterative'
        # will store extracted components here
        components = []
        
        # there are as many components as there are dimensions
        for dimension in range(dimensions):
            # make an initial guess
            old_guess = np.random.uniform(-1.,1.,dimensions)

            # project out any component along previously extracted components
            old_guess = old_guess - sum( 
                [
                    component.transpose().dot(old_guess) * component for component in components
                ],
                np.zeros(dimensions)
            )

            # normalize
            old_guess = old_guess / np.linalg.norm(old_guess)

            # keep track of number of iterations it takes to converge
            iterations = 0
            while True:
                iterations += 1

                # compute improved component from old one
                new_guess = np.average(
                    whitened * F(
                        old_guess.transpose().dot(whitened)
                    ).reshape(1,samples),
                    1
                ) - np.average(
                    f(
                        old_guess.transpose().dot(whitened)
                    )            
                ) * old_guess

                # perform same projection / normalization as we did with first guess
                new_guess = new_guess - sum( 
                    [
                        component.transpose().dot(new_guess) * component for component in components
                    ],
                    np.zeros(dimensions)
                )
                new_guess = new_guess / np.linalg.norm(new_guess)

                # compute difference between new and old guess
                delta_pos = np.linalg.norm(new_guess - old_guess)

                # compute difference between new and negative of old guess
                delta_neg = np.linalg.norm(new_guess + old_guess)

                # set new guess to be old guess of next loop
                old_guess = new_guess

                # if old guess is "same" as new guess (i.e. within arbitrary negative sign) then we're done
                if delta_pos < delta or delta_neg < delta:
                    break        
            # add extracted component to list
            components.append(old_guess)
            print 'dimension %d found on %d iterations' % (dimension + 1,iterations)

    # Symmetric calculates all parameters together using lin alg
    if method == 'symmetric':
        print 'method is symmetric'

        # will store extracted components here
        components = np.zeros((dimensions,dimensions))
        newComponent = np.zeros((dimensions,dimensions))
    

        for dimension in range(dimensions):
            old_guess = np.random.uniform(-1.,1.,dimensions)
            components[dimension,:] = old_guess

        s, u = np.linalg.eigh(np.dot(components, components.T))
        components = np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), components)

        # components = components/ np.linalg.norm(components)

        iterations = 0

        while True:
            
            iterations+=1
            newComponent = np.zeros((dimensions,dimensions))
            
            for dimension in range(dimensions):
                old_guess = components[dimension,:]
                
                new_guess = np.average(
                    whitened * F(
                        old_guess.transpose().dot(whitened)
                    ).reshape(1,samples),
                    1
                ) - np.average(
                    f(
                        old_guess.transpose().dot(whitened)
                    )            
                ) * old_guess


                
                newComponent[dimension,:] = new_guess

            print newComponent
            # Symetric Decorrelation
            s, u = np.linalg.eigh(np.dot(newComponent, newComponent.T))
            newComponent = np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), newComponent)

            #print "OldS \n", components, "\n NewS \n", newComponent
            lim = max(np.abs(np.abs(np.diag(newComponent.dot(components.T)))-1))

            # set new guess to be old guess of next loop
            components = newComponent


            # if old guess is "same" as new guess (i.e. within arbitrary negative sign) then we're done
            if lim < delta:
                break      
            print '%d iterations' % (iterations), lim

    # Symmetric_approx calculates simmilar to Symmetric but uses approx instead of lin alg
    if method == symmetric_approx:
        pass

    # compute extracted signals by applying extracted weights on whitened signal data
    extracted_signals = np.vstack(components).dot(whitened)
    return extracted_signals

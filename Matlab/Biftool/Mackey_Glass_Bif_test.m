%% Define name, dimensions, and path

addpath(path, '/Users/michael/Documents/Biftool/dde_biftool_v3.1.1/ddebiftool',...
    '/Users/michael/Documents/Biftool/dde_biftool_v3.1.1/ddebiftool_extra_psol',...
    '/Users/michael/Documents/Biftool/dde_biftool_v3.1.1/ddebiftool_extra_nmfm',...
    '/Users/michael/Documents/Biftool/dde_biftool_v3.1.1/ddebiftool_utilities')

%% Enable Vectorization

x_vectorize = true;

%% Set user-defined functions
gamma = 1.0;
beta_ind = 1;   %sref. to value of beta in array defining parameters
n_ind = 2;
tau_ind = 3;

% p = [2,7,2];

if x_vectorize
    f=@(x,xtau,beta,n)beta*xtau./(1+xtau.^n)-gamma*x;
    funcs=set_funcs(...
        'sys_rhs',@(xx,p)f(xx(1,1,:),xx(1,2,:),p(1),p(2)),...
        'sys_tau',@()tau_ind,...
        'x_vectorized',true);
else
    f=@(x,xtau,beta,n)beta*xtau/(1+xtau^n)-gamma*x; %#ok<UNRCH>
    funcs=set_funcs(...
        'sys_rhs',@(xx,p)f(xx(1,1,:),xx(1,2,:),p(1),p(2)),...
        'sys_tau',@()p(tau_ind));
end
%% Initial parameters and state
beta0=2;
n0=10;
tau0=0;
x0=(beta0-1)^(1/n0);
%% Initialization of branch of non-trivial equilibria
contpar=tau_ind;
nontriv_eqs=SetupStst(funcs,'x',x0,'parameter',[beta0,n0,tau0],'step',0.1,...
    'contpar',contpar,'max_step',[contpar,0.3],'max_bound',[contpar,10]);
%% Compute and find stability of non-trivial equilibria 
disp('Trivial equilibria');
figure(1);clf
nontriv_eqs=br_contn(funcs,nontriv_eqs,3);
nontriv_eqs=br_stabl(funcs,nontriv_eqs,0,1);
nunst_eqs=GetStability(nontriv_eqs);
ind_hopf=find(nunst_eqs<2,1,'last');
fprintf('Hopf bifurcation near point %d\n',ind_hopf);